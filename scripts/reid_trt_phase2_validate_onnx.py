"""
Phase 2: OSNet ONNX Output Validation
=======================================
Runs a thorough comparison between PyTorch eager mode and ONNX Runtime
to ensure the OSNet export is correct before TensorRT conversion.

Tests:
  1. Raw feature output match (batch=1, zero input)
  2. Raw feature output match (batch=1, synthetic crop)
  3. L2-normalized embedding parity (cosine similarity)
  4. Same-person similarity is preserved (sim_A1_A2 > sim_A_B)
  5. Throughput comparison (PyTorch eager vs ONNX Runtime)

Usage:
    cd <project_root>
    python scripts/reid_trt_phase2_validate_onnx.py
"""

import sys
import os

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ONNX_PATH  = os.path.join(MODELS_DIR, "osnet_x1_0_b1.onnx")
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

# Preprocessing constants (must match embedding_extractor.py)
_INPUT_H = 128
_INPUT_W = 128
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_crop_np(crop_bgr: np.ndarray) -> np.ndarray:
    """
    CPU numpy preprocessing — mirrors _preprocess_batch_fast internals.
    Input:  (H, W, 3) uint8 BGR — already resized to 128×128
    Output: (1, 3, 128, 128) float32, ImageNet normalized
    """
    import cv2
    # Ensure 128x128
    if crop_bgr.shape[:2] != (_INPUT_H, _INPUT_W):
        crop_bgr = cv2.resize(crop_bgr, (_INPUT_W, _INPUT_H))
    rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB, [0,1]
    rgb = (rgb - _MEAN) / _STD                              # ImageNet normalize
    chw = rgb.transpose(2, 0, 1)[np.newaxis]               # (1, 3, H, W)
    return np.ascontiguousarray(chw)


def make_synthetic_crops():
    """Generate reproducible synthetic person crops for testing."""
    rng = np.random.RandomState(42)
    # person_a: warm-toned
    person_a  = np.full((_INPUT_H, _INPUT_W, 3), [50, 80, 200], dtype=np.uint8)
    noise     = rng.randint(-15, 15, person_a.shape, dtype=np.int16)
    person_a2 = np.clip(person_a.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # person_b: cool-toned (distinct)
    person_b  = np.full((_INPUT_H, _INPUT_W, 3), [200, 50, 50], dtype=np.uint8)
    return person_a, person_a2, person_b


# ══════════════════════════════════════════════════════════════════════════════
#  Load models
# ══════════════════════════════════════════════════════════════════════════════

def load_pytorch_osnet() -> torch.nn.Module:
    """Load OSNet via EmbeddingExtractor (uses existing weight loading logic)."""
    from reid.embedding_extractor import EmbeddingExtractor
    extractor = EmbeddingExtractor(half=False)   # FP32 for comparison purity
    # Return the raw nn.Module (used in ONNX wrapper)
    return extractor._model, extractor


def load_ort_session(onnx_path: str):
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "cuda" in DEVICE else ["CPUExecutionProvider"])
    sess_opts = ort.SessionOptions()
    sess_opts.enable_mem_pattern = True
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    print(f"[VALIDATE] ORT providers: {sess.get_providers()}")
    return sess


# ══════════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_pytorch(model: torch.nn.Module, batch_np: np.ndarray) -> np.ndarray:
    """Run batch through PyTorch OSNet (raw features, no L2)."""
    with torch.no_grad():
        t = torch.from_numpy(batch_np).to(DEVICE)
        feats = model(t)
        return feats.float().cpu().numpy()


def run_ort(sess, batch_np: np.ndarray) -> np.ndarray:
    """Run batch through ONNX Runtime."""
    return sess.run(None, {"images": batch_np})[0]


def l2_normalize(feats: np.ndarray) -> np.ndarray:
    """L2-normalize rows of (N, D) array."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized 1D vectors."""
    return float(np.dot(a, b))


# ══════════════════════════════════════════════════════════════════════════════
#  Test 1: Zero input — batch=1
# ══════════════════════════════════════════════════════════════════════════════

def test_zero_input(model, sess) -> bool:
    batch = np.zeros((1, 3, _INPUT_H, _INPUT_W), dtype=np.float32)
    pt  = run_pytorch(model, batch)
    ort = run_ort(sess, batch)
    err = np.abs(pt - ort).max()
    ok  = err < 1e-3
    status = "✓ OK" if ok else "✗ FAIL"
    print(f"  [{status}] Zero input (batch=1)  — max_err={err:.2e}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Test 2: Synthetic crop — raw feature comparison
# ══════════════════════════════════════════════════════════════════════════════

def test_synthetic_crop(model, sess) -> bool:
    person_a, _, _ = make_synthetic_crops()
    batch = preprocess_crop_np(person_a)
    pt  = run_pytorch(model, batch)
    ort = run_ort(sess, batch)
    err = np.abs(pt - ort).max()
    ok  = err < 1e-3
    status = "✓ OK" if ok else "✗ FAIL"
    print(f"  [{status}] Synthetic crop (batch=1) — max_err={err:.2e}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Test 3: Embedding shape and L2-norm validity
# ══════════════════════════════════════════════════════════════════════════════

def test_embedding_shape_and_norm(model, sess) -> bool:
    person_a, _, _ = make_synthetic_crops()
    batch = preprocess_crop_np(person_a)
    ort_feats = run_ort(sess, batch)
    emb       = l2_normalize(ort_feats)[0]

    shape_ok = ort_feats.shape == (1, 512)
    norm_ok  = abs(np.linalg.norm(emb) - 1.0) < 1e-5

    ok = shape_ok and norm_ok
    status = "✓ OK" if ok else "✗ FAIL"
    print(f"  [{status}] Embedding shape={ort_feats.shape}  "
          f"L2-norm={np.linalg.norm(emb):.6f} (expected 1.0)")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Test 4: Identity preservation — same-person similarity vs different-person
# ══════════════════════════════════════════════════════════════════════════════

def test_identity_preservation(model, sess) -> bool:
    person_a, person_a2, person_b = make_synthetic_crops()

    batch_a  = preprocess_crop_np(person_a)
    batch_a2 = preprocess_crop_np(person_a2)
    batch_b  = preprocess_crop_np(person_b)

    emb_a  = l2_normalize(run_ort(sess, batch_a))[0]
    emb_a2 = l2_normalize(run_ort(sess, batch_a2))[0]
    emb_b  = l2_normalize(run_ort(sess, batch_b))[0]

    sim_same = cosine_sim(emb_a, emb_a2)
    sim_diff = cosine_sim(emb_a, emb_b)
    ok = sim_same > sim_diff

    status = "✓ OK" if ok else "✗ FAIL"
    print(f"  [{status}] Identity preservation — "
          f"sim_same={sim_same:.4f}  sim_diff={sim_diff:.4f}  "
          f"margin={sim_same - sim_diff:.4f}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Test 5: PyTorch vs ORT embedding cosine agreement
# ══════════════════════════════════════════════════════════════════════════════

def test_cosine_agreement(model, sess) -> bool:
    """
    Verify that PyTorch and ORT embeddings point in the same direction
    (high cosine similarity) even if raw values diverge slightly due to
    FP arithmetic differences.
    """
    person_a, _, _ = make_synthetic_crops()
    batch = preprocess_crop_np(person_a)

    pt_emb  = l2_normalize(run_pytorch(model, batch))[0]
    ort_emb = l2_normalize(run_ort(sess, batch))[0]

    sim = cosine_sim(pt_emb, ort_emb)
    ok  = sim > 0.9999   # Should be essentially identical (FP32)
    status = "✓ OK" if ok else "✗ FAIL"
    print(f"  [{status}] PyTorch vs ORT cosine similarity: {sim:.8f} (expected >0.9999)")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  Test 6: Throughput comparison
# ══════════════════════════════════════════════════════════════════════════════

def test_throughput(model, sess, n_runs: int = 50):
    import time

    batch = preprocess_crop_np(
        np.full((_INPUT_H, _INPUT_W, 3), 128, dtype=np.uint8)
    )

    # PyTorch warmup + bench
    for _ in range(10):
        run_pytorch(model, batch)
    if "cuda" in DEVICE:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_pytorch(model, batch)
    if "cuda" in DEVICE:
        torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / n_runs * 1000

    # ORT warmup + bench
    for _ in range(10):
        run_ort(sess, batch)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_ort(sess, batch)
    ort_ms = (time.perf_counter() - t0) / n_runs * 1000

    speedup = pt_ms / max(ort_ms, 0.001)
    print(f"\n  Throughput (batch=1, {n_runs} runs):")
    print(f"    PyTorch eager : {pt_ms:.2f} ms/inference")
    print(f"    ONNX Runtime  : {ort_ms:.2f} ms/inference  ({speedup:.2f}x vs PyTorch)")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SELFWATCH — OSNet ONNX Validation (Phase 2)")
    print("=" * 60)

    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] ONNX not found: {ONNX_PATH}")
        print("[ERROR] Run Phase 1 first: python scripts/reid_trt_phase1_export_onnx.py")
        sys.exit(1)

    try:
        import onnxruntime as ort
        print(f"[VALIDATE] ORT version : {ort.__version__}")
    except ImportError:
        print("[ERROR] onnxruntime not installed: pip install onnxruntime-gpu")
        sys.exit(1)

    print(f"[VALIDATE] ONNX path   : {ONNX_PATH}")
    print(f"[VALIDATE] Device      : {DEVICE}")
    print()

    print("[VALIDATE] Loading PyTorch OSNet …")
    nn_model, extractor = load_pytorch_osnet()
    print("[VALIDATE] Loading ORT session …")
    sess = load_ort_session(ONNX_PATH)

    print("\n[VALIDATE] Running tests:\n")
    results = [
        test_zero_input(nn_model, sess),
        test_synthetic_crop(nn_model, sess),
        test_embedding_shape_and_norm(nn_model, sess),
        test_identity_preservation(nn_model, sess),
        test_cosine_agreement(nn_model, sess),
    ]
    test_throughput(nn_model, sess)

    passed = sum(results)
    total  = len(results)
    print(f"\n[VALIDATE] {passed}/{total} tests passed")
    if passed == total:
        print("[VALIDATE] ✓ ONNX export is valid — safe to proceed to Phase 3 (TensorRT).")
    else:
        print("[VALIDATE] ✗ Fix ONNX issues before converting to TensorRT.")
        sys.exit(1)
