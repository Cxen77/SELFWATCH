"""
Phase 5: OSNet TensorRT vs PyTorch Embedding Validation
=========================================================
Validates that the TensorRT FP16 conversion preserves embedding
quality and identity matching characteristics.

Checks:
1. Single-image and Batched (1,4,8,16) embedding parity
2. Numerical precision (Cosine, L2, Max Abs Error, Mean Abs Error)
3. Similarity preservation (Same-person vs Different-person pairs)
4. ReID threshold stability and Cross-Camera separation margin

Acceptance Criteria:
- Mean cosine similarity between PT/TRT > 0.999
- Minimum cosine similarity between PT/TRT > 0.995
- TRT separation margin within 2% of PyTorch margin

Usage:
    cd <project_root>
    python scripts/reid_trt_phase5_embedding_validation.py
"""

import sys
import os
import numpy as np
import cv2

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from reid.embedding_extractor import EmbeddingExtractor
from reid.trt_embedding_extractor import TRTEmbeddingExtractor

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
ENGINE_PATH = os.path.join(MODELS_DIR, "osnet_x1_0_dyn_fp16.engine")

# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic Image Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_test_crops(n_identities=5, crops_per_id=4):
    """
    Generate synthetic crops representing different "identities" 
    under different "camera conditions".
    """
    rng = np.random.RandomState(42)
    crops = []
    labels = []
    
    for identity_id in range(n_identities):
        # Base appearance for this identity
        base_color = rng.randint(0, 255, size=3).astype(np.uint8)
        base_crop = np.full((128, 128, 3), base_color, dtype=np.uint8)
        
        # Add random shapes/patterns to make it unique but structured
        for _ in range(5):
            x = rng.randint(0, 100)
            y = rng.randint(0, 100)
            w = rng.randint(10, 30)
            h = rng.randint(10, 30)
            c = rng.randint(0, 255, size=3).tolist()
            cv2.rectangle(base_crop, (x, y), (x+w, y+h), c, -1)
            
        for view_idx in range(crops_per_id):
            # Simulate different views/lighting (noise, brightness shifts)
            crop = base_crop.copy().astype(np.int16)
            noise = rng.randint(-20, 20, size=(128, 128, 3))
            brightness = rng.randint(-15, 15)
            crop = np.clip(crop + noise + brightness, 0, 255).astype(np.uint8)
            
            crops.append(crop)
            labels.append(identity_id)
            
    return crops, labels

# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)

def compute_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)

def compute_errors(pt: np.ndarray, trt: np.ndarray):
    diff = np.abs(pt - trt)
    return diff.max(), diff.mean()

# ══════════════════════════════════════════════════════════════════════════════
#  Main Validation
# ══════════════════════════════════════════════════════════════════════════════

def run_validation():
    print("=" * 70)
    print("  SELFWATCH — Phase 5: TRT Embedding Validation")
    print("=" * 70)

    if not os.path.exists(ENGINE_PATH):
        print(f"[ERROR] Engine not found: {ENGINE_PATH}")
        sys.exit(1)

    print("[VALIDATE] Loading PyTorch Baseline …")
    pt_ext = EmbeddingExtractor(half=True)
    
    print("[VALIDATE] Loading TensorRT Engine …")
    trt_ext = TRTEmbeddingExtractor(engine_path=ENGINE_PATH, fallback=False)

    crops, labels = generate_test_crops(n_identities=10, crops_per_id=5)
    
    # ── 1. & 2. Batch Embedding Parity ──────────────────────────────────────
    print("\n[TEST] 1 & 2. Single-image and Batch Embedding Parity")
    
    all_cosines = []
    max_dev = 0.0
    
    for bs in [1, 4, 8, 16]:
        # Extract a batch of size bs
        batch_crops = crops[:bs]
        
        pt_embs = pt_ext.extract_batch(batch_crops)
        trt_embs = trt_ext.extract_batch(batch_crops)
        
        cos_sims = compute_cosine(pt_embs, trt_embs)
        l2_dists = compute_l2(pt_embs, trt_embs)
        max_err, mean_err = compute_errors(pt_embs, trt_embs)
        
        all_cosines.extend(cos_sims.tolist())
        max_dev = max(max_dev, max_err)
        
        print(f"  Batch={bs:<2} | Cosine: {cos_sims.mean():.6f} (min: {cos_sims.min():.6f}) | "
              f"L2: {l2_dists.mean():.4f} | MaxErr: {max_err:.1e}")

    # ── 3. & 4. Similarity Preservation and Threshold Stability ──────────────
    print("\n[TEST] 3 & 4. Similarity Preservation & Threshold Stability")
    
    # Extract all crops
    pt_all = pt_ext.extract_batch(crops)
    trt_all = trt_ext.extract_batch(crops)
    
    pt_same, trt_same = [], []
    pt_diff, trt_diff = [], []
    
    n_total = len(crops)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            pt_sim = float(compute_cosine(pt_all[i], pt_all[j]))
            trt_sim = float(compute_cosine(trt_all[i], trt_all[j]))
            
            if labels[i] == labels[j]:
                pt_same.append(pt_sim)
                trt_same.append(trt_sim)
            else:
                pt_diff.append(pt_sim)
                trt_diff.append(trt_sim)

    # ── 5. Numerical Precision Report ───────────────────────────────────────
    mean_cos = np.mean(all_cosines)
    min_cos  = np.min(all_cosines)
    
    print("\n[TEST] 5. Numerical Precision Report")
    print(f"  Mean Cosine Similarity : {mean_cos:.6f}")
    print(f"  Min Cosine Similarity  : {min_cos:.6f}")
    print(f"  Max Embedding Deviation: {max_dev:.6f}")

    # ── 6. Cross-camera identity stability test ─────────────────────────────
    print("\n[TEST] 6. Identity Separation Margin")
    
    pt_same_mean = np.mean(pt_same)
    pt_diff_mean = np.mean(pt_diff)
    pt_margin = pt_same_mean - pt_diff_mean
    
    trt_same_mean = np.mean(trt_same)
    trt_diff_mean = np.mean(trt_diff)
    trt_margin = trt_same_mean - trt_diff_mean
    
    margin_diff_pct = abs(pt_margin - trt_margin) / pt_margin * 100

    print("  PyTorch Baseline:")
    print(f"    Same-person mean : {pt_same_mean:.4f}")
    print(f"    Diff-person mean : {pt_diff_mean:.4f}")
    print(f"    Separation Margin: {pt_margin:.4f}")
    
    print("  TensorRT Backend:")
    print(f"    Same-person mean : {trt_same_mean:.4f}")
    print(f"    Diff-person mean : {trt_diff_mean:.4f}")
    print(f"    Separation Margin: {trt_margin:.4f}")
    
    print(f"\n  Margin Difference  : {margin_diff_pct:.4f}%")

    # ── Acceptance Evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    
    crit_1 = mean_cos > 0.999
    crit_2 = min_cos > 0.995
    crit_3 = margin_diff_pct <= 2.0
    
    print(f"  [ {'PASS' if crit_1 else 'FAIL'} ] Mean cosine > 0.999 ({mean_cos:.6f})")
    print(f"  [ {'PASS' if crit_2 else 'FAIL'} ] Min cosine  > 0.995 ({min_cos:.6f})")
    print(f"  [ {'PASS' if crit_3 else 'FAIL'} ] Margin diff <= 2.0% ({margin_diff_pct:.4f}%)")
    
    print("\n  Recommendation:")
    if crit_1 and crit_2 and crit_3:
        print("  >>> SAFE TO DEPLOY <<<")
        print("  TensorRT embeddings are mathematically equivalent for ReID tracking.")
        print("  Existing thresholds do not require retuning.")
    else:
        print("  >>> INVESTIGATE FURTHER <<<")
        print("  TensorRT conversion resulted in significant embedding degradation.")
        print("  Do not deploy. Check FP16 precision issues or ONNX node fusion.")

if __name__ == "__main__":
    run_validation()
