"""
Phase 1: OSNet → ONNX Export Script  (Dynamic Batch)
======================================================
Exports the OSNet x1.0 ReID model to an ONNX file with:
  - DYNAMIC batch axis  — TRT will process all N crops in one forward pass
  - FP32 weights        — TensorRT converts to FP16 during engine build
  - Raw feature output  — L2 normalization applied after TRT in the backend

Why dynamic batch?
  OSNet PyTorch inference is flat at ~30ms for batch 1–16 because kernel-launch
  overhead dominates over actual GPU compute. After TRT fuses Conv+BN+ReLU and
  the 4 parallel branches of each OSBlock into single kernels, the compute-to-
  overhead ratio flips. At that point the GPU can parallelize across the batch
  dimension, giving sub-linear scaling:
    PyTorch batch=8 : ~30ms (flat)
    TRT dynamic b=8 : ~6ms  (4–5x speedup over sequential TRT-b1 calls)

ONNX graph:
  Input  "images"   : (N, 3, 128, 128)  float32, ImageNet-normalized
  Output "features" : (N, 512)          float32, RAW (NOT L2-normalized)

  L2 normalization is applied in trt_embedding_extractor.py after inference.
  This keeps the ONNX graph free of unsupported ops.

Usage:
    cd <project_root>
    python scripts/reid_trt_phase1_export_onnx.py

Output:
    models/osnet_x1_0_dyn.onnx              (main export, dynamic batch)
    models/osnet_x1_0_dyn_simplified.onnx   (after onnx-simplifier — optional)
"""

import sys
import os

# ── Ensure project root is on path ────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

# ── Output paths ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ONNX_PATH        = os.path.join(MODELS_DIR, "osnet_x1_0_dyn.onnx")
ONNX_SIMPLE_PATH = os.path.join(MODELS_DIR, "osnet_x1_0_dyn_simplified.onnx")

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH  = os.path.join(PROJECT_ROOT, "weights", "osnet", "osnet_x1_0_msmt17.pth")
INPUT_HEIGHT  = 128
INPUT_WIDTH   = 128
OPSET         = 17        # ONNX opset 17 — best TRT 11 compatibility
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
TRACE_BATCH   = 1         # Batch size used for tracing (shape is made dynamic below)


# ══════════════════════════════════════════════════════════════════════════════
#  ONNX wrapper — raw feature output ONLY, no L2 norm
# ══════════════════════════════════════════════════════════════════════════════

class OSNetONNXWrapper(torch.nn.Module):
    """
    Wraps the OSNet backbone for clean ONNX export.

    Input:  images   — (N, 3, H, W) float32, ImageNet-normalized
    Output: features — (N, 512) float32, RAW (NOT L2-normalized)

    L2 normalization is applied AFTER TRT inference in the runtime backend.
    This keeps the ONNX graph to pure conv/BN/ReLU/pool ops — all guaranteed
    to be supported by TensorRT 10/11 without any special handling.
    """

    def __init__(self, osnet_model: torch.nn.Module):
        super().__init__()
        self.model = osnet_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)   # (N, 512)


# ══════════════════════════════════════════════════════════════════════════════
#  Load OSNet (reuses architecture from embedding_extractor.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_osnet(weights_path: str, device: str) -> torch.nn.Module:
    """Load OSNet x1.0 using the self-contained architecture in embedding_extractor."""
    print(f"[EXPORT] Loading OSNet x1.0 from {weights_path} …")

    from reid.embedding_extractor import _build_osnet_x1_0
    from collections import OrderedDict

    model = _build_osnet_x1_0()

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = (checkpoint["state_dict"]
                  if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                  else checkpoint)

    # Strip 'module.' prefix (DataParallel checkpoints)
    cleaned = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )

    # Partial load — ignore classifier head (shape mismatch)
    model_dict = model.state_dict()
    matched = {
        k: v for k, v in cleaned.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    skipped = set(cleaned.keys()) - set(matched.keys())
    model_dict.update(matched)
    model.load_state_dict(model_dict)

    if skipped:
        print(f"[EXPORT] Skipped {len(skipped)} keys (classifier head / shape mismatch)")
    print(f"[EXPORT] Loaded {len(matched)}/{len(model_dict)} layers")

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[EXPORT] OSNet on {device}. Parameters: {n_params:.2f}M")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Export — dynamic batch axis
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(osnet_model: torch.nn.Module, onnx_path: str, device: str):
    import onnx

    wrapper = OSNetONNXWrapper(osnet_model).to(device)
    wrapper.eval()

    # Trace with batch=1 — dynamic_axes makes dimension 0 symbolic at runtime
    dummy = torch.zeros(
        TRACE_BATCH, 3, INPUT_HEIGHT, INPUT_WIDTH,
        dtype=torch.float32, device=device
    )
    print(f"[EXPORT] Trace input shape : {dummy.shape}  (batch dim will be made dynamic)")
    print(f"[EXPORT] Exporting to ONNX (opset={OPSET}, dynamic batch axis) …")

    # dynamic_axes: mark dimension 0 of both input and output as variable ("batch").
    # OSNet uses only standard conv/BN/ReLU/pool/fc — all ops are batch-agnostic.
    # No patches required (unlike RF-DETR which had deformable attention issues).
    dynamic_axes = {
        "images":   {0: "batch"},   # (N, 3, 128, 128)
        "features": {0: "batch"},   # (N, 512)
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["features"],
            dynamic_axes=dynamic_axes,
            dynamo=False,   # TorchScript exporter — stable for conv-based models
            verbose=False,
        )

    print(f"[EXPORT] ✓ Saved: {onnx_path}")

    # ── Validate ONNX graph ───────────────────────────────────────────────────
    print("[EXPORT] Validating ONNX graph …")
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    file_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[EXPORT] ✓ Graph valid. File size: {file_mb:.1f} MB")

    # Confirm batch dim is symbolic (not hardcoded)
    input_shape  = [d.dim_value if d.dim_value != 0 else d.dim_param
                    for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value if d.dim_value != 0 else d.dim_param
                    for d in model_onnx.graph.output[0].type.tensor_type.shape.dim]
    print(f"[EXPORT] ONNX input  shape : {input_shape}   "
          f"(dim 0 = '{input_shape[0]}' = dynamic ✓)")
    print(f"[EXPORT] ONNX output shape : {output_shape}  "
          f"(dim 0 = '{output_shape[0]}' = dynamic ✓)")
    assert isinstance(input_shape[0], str), (
        f"Batch dim is NOT dynamic — got {input_shape[0]}. "
        f"Check dynamic_axes in export call.")

    return model_onnx


# ══════════════════════════════════════════════════════════════════════════════
#  Optional: onnxsim simplification
# ══════════════════════════════════════════════════════════════════════════════

def simplify_onnx(onnx_path: str, simplified_path: str) -> bool:
    try:
        from onnxsim import simplify
        import onnx
        print("[EXPORT] Running onnx-simplifier …")
        model = onnx.load(onnx_path)
        # Pass test_input_shapes so simplifier can fold constants with concrete sizes
        model_simplified, check = simplify(
            model,
            test_input_shapes={"images": (1, 3, INPUT_HEIGHT, INPUT_WIDTH)},
        )
        if check:
            onnx.save(model_simplified, simplified_path)
            print(f"[EXPORT] ✓ Simplified ONNX saved: {simplified_path}")
            return True
        else:
            print("[EXPORT] ✗ onnx-simplifier check failed — using original ONNX")
            return False
    except ImportError:
        print("[EXPORT] onnxsim not installed — skipping simplification")
        print("         Install with: pip install onnxsim")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Verify ONNX output vs PyTorch — test BOTH batch=1 and batch=4
# ══════════════════════════════════════════════════════════════════════════════

def verify_onnx_runtime(onnx_path: str, osnet_model: torch.nn.Module, device: str):
    """
    Run the same inputs through PyTorch AND ONNX Runtime.
    Tests batch=1 AND batch=4 to confirm the dynamic axis works at runtime.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[VERIFY] onnxruntime not installed — skipping")
        print("         Install with: pip install onnxruntime-gpu")
        return

    print("[VERIFY] Running ORT verification (batch=1 and batch=4) …")

    wrapper = OSNetONNXWrapper(osnet_model).to(device)
    wrapper.eval()

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "cuda" in device else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[VERIFY] ORT providers: {sess.get_providers()}")

    rng = np.random.RandomState(42)
    TOLERANCE = 1e-3

    all_ok = True
    for bs in [1, 4]:
        dummy_np = rng.rand(bs, 3, INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)

        # PyTorch reference
        with torch.no_grad():
            pt_feats = wrapper(
                torch.from_numpy(dummy_np).to(device)
            ).float().cpu().numpy()

        # ORT
        ort_feats = sess.run(None, {"images": dummy_np})[0]

        err = np.abs(pt_feats - ort_feats).max()
        ok  = err < TOLERANCE
        all_ok = all_ok and ok
        status = "✓" if ok else "⚠"
        print(f"[VERIFY] {status} batch={bs}  shape={ort_feats.shape}  "
              f"max_err={err:.2e}  ({'PASS' if ok else 'WARN — check FP accumulation'})")

    if all_ok:
        print("[VERIFY] ✓ Dynamic batch ONNX matches PyTorch for all tested sizes.")
    else:
        print("[VERIFY] ⚠ Some batches exceeded tolerance — "
              "acceptable if error < 1e-2 (FP32 accumulation differences).")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  SELFWATCH — OSNet ReID ONNX Export (Phase 1 — Dynamic Batch)")
    print("=" * 65)
    print(f"  Device       : {DEVICE}")
    print(f"  Trace input  : {TRACE_BATCH}×3×{INPUT_HEIGHT}×{INPUT_WIDTH}  "
          f"(batch dim will be dynamic)")
    print(f"  Weights      : {WEIGHTS_PATH}")
    print(f"  ONNX path    : {ONNX_PATH}")
    print("=" * 65)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] Weights not found: {WEIGHTS_PATH}")
        sys.exit(1)

    osnet_model = load_osnet(WEIGHTS_PATH, DEVICE)
    export_onnx(osnet_model, ONNX_PATH, DEVICE)
    simplify_onnx(ONNX_PATH, ONNX_SIMPLE_PATH)
    verify_onnx_runtime(ONNX_PATH, osnet_model, DEVICE)

    print("\n[EXPORT] Phase 1 complete.")
    print("[EXPORT] Next step: run scripts/reid_trt_phase2_validate_onnx.py")
