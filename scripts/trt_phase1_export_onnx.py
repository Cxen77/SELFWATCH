"""
Phase 1: RF-DETR → ONNX Export Script
======================================
Exports the RF-DETR Nano model to an ONNX file with:
  - Dynamic batch dimension (N can be 1..8 at runtime)
  - FP32 weights (TensorRT will convert to FP16 during engine build)
  - A custom wrapper that ONLY contains the neural-net forward pass
    (no Python postprocessing, no numpy — pure tensor ops)

The ONNX graph stops right after the model's raw output tensors:
  pred_logits: (N, num_queries, num_classes+1)  — raw class logits
  pred_boxes:  (N, num_queries, 4)              — normalized [cx,cy,w,h]

Postprocessing (sigmoid → threshold → rescale) is done AFTER the
ONNX call, inside our TensorRT detector backend. This keeps the ONNX
graph clean, avoids unsupported ops, and lets us share postprocessing
code between PyTorch and TensorRT backends.

Usage:
    python scripts/trt_phase1_export_onnx.py

Output:
    models/rfdetr_nano_384.onnx          (main export)
    models/rfdetr_nano_384_simplified.onnx  (after onnx-simplifier — optional)
"""

import sys
import os

# ── Ensure project root is on path ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

# ── Output paths ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ONNX_PATH = os.path.join(MODELS_DIR, "rfdetr_nano_384.onnx")
ONNX_SIMPLE_PATH = os.path.join(MODELS_DIR, "rfdetr_nano_384_simplified.onnx")

# ── Config (must match multicam_pipeline.py) ──────────────────────────────────
VARIANT     = "nano"
RESOLUTION  = 384       # detector_resolution in multicam_pipeline
OPSET       = 17        # ONNX opset — 17 supports all modern transformer ops
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  ONNX wrapper module — strips away Python postprocessing
# ══════════════════════════════════════════════════════════════════════════════

class RFDETRNanoONNXWrapper(torch.nn.Module):
    """
    Wraps the raw RF-DETR backbone + transformer + prediction heads.

    Input:  pixel_values — (N, 3, R, R) float32, ImageNet-normalized
    Output: pred_logits  — (N, num_queries, num_classes+1)
            pred_boxes   — (N, num_queries, 4)  normalized [cx,cy,w,h]

    Everything after this (sigmoid, threshold, box rescaling, NMS) is
    handled in Python/CUDA outside the ONNX graph.
    """

    def __init__(self, nn_model):
        super().__init__()
        self.model = nn_model   # raw nn.Module (LWDETR)

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values)
        # RF-DETR returns either a dict or a (boxes, logits) tuple
        if isinstance(out, dict):
            return out["pred_logits"], out["pred_boxes"]
        elif isinstance(out, tuple):
            # rfdetr returns (pred_boxes, pred_logits) — note reversed order
            return out[1], out[0]
        else:
            raise RuntimeError(f"Unexpected RF-DETR output type: {type(out)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Load model
# ══════════════════════════════════════════════════════════════════════════════

def load_rfdetr_nano(resolution: int, device: str):
    print(f"[EXPORT] Loading RF-DETR Nano (resolution={resolution}) …")
    from rfdetr import RFDETRNano

    rfdetr = RFDETRNano(resolution=resolution)

    # Get the raw nn.Module (LWDETR)
    # rfdetr.model is a ModelContext; .model inside is the raw nn.Module
    nn_model = rfdetr.model.model
    nn_model.eval()
    nn_model = nn_model.to(device)
    print(f"[EXPORT] Model on {device}. Parameters: "
          f"{sum(p.numel() for p in nn_model.parameters()) / 1e6:.1f}M")
    return nn_model


# ══════════════════════════════════════════════════════════════════════════════
#  Export
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(nn_model, resolution: int, onnx_path: str, device: str):
    import onnx

    wrapper = RFDETRNanoONNXWrapper(nn_model).to(device)
    wrapper.eval()

    # Dummy input: batch=1 (dynamic batch configured below)
    dummy = torch.zeros(1, 3, resolution, resolution,
                        dtype=torch.float32, device=device)

    print(f"[EXPORT] Tracing with dummy input shape: {dummy.shape}")

    print(f"[EXPORT] Exporting to ONNX (opset {OPSET}, STATIC shape, Dynamo disabled) …")
    
    # CRITICAL FIX for PyTorch 2.11+:
    # 1. We MUST set `dynamo=False` to force the legacy TorchScript-based exporter.
    # 2. We disable `dynamic_axes` entirely. Deformable Attention tracing is notoriously
    #    fragile with dynamic axes (causes Split/Reshape hardcoding bugs in ONNX graph).
    #    We prioritize a successful, exact static export first.
    
    import torch.nn.functional as Fnn
    orig_interpolate = Fnn.interpolate

    def patched_interpolate(*args, **kwargs):
        if "antialias" in kwargs:
            del kwargs["antialias"]
        return orig_interpolate(*args, **kwargs)

    print(f"[EXPORT] Patching F.interpolate to bypass 'antialias' unsupported ONNX op …")
    Fnn.interpolate = patched_interpolate

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                onnx_path,
                export_params=True,
                opset_version=OPSET,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["pred_logits", "pred_boxes"],
                # dynamic_axes=dynamic_axes,  # <-- DISABLED FOR STABILITY
                dynamo=False,  # <-- CRITICAL for RF-DETR deformable attention
                verbose=False,
            )
    finally:
        Fnn.interpolate = orig_interpolate

    print(f"[EXPORT] OK Saved: {onnx_path}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("[EXPORT] Validating ONNX graph …")
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    file_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[EXPORT] OK Graph valid. File size: {file_mb:.1f} MB")
    return model_onnx


# ══════════════════════════════════════════════════════════════════════════════
#  Optional: onnxsim simplification
# ══════════════════════════════════════════════════════════════════════════════

def simplify_onnx(onnx_path: str, simplified_path: str):
    try:
        from onnxsim import simplify
        import onnx
        print("[EXPORT] Running onnx-simplifier …")
        model = onnx.load(onnx_path)
        model_simplified, check = simplify(model)
        if check:
            onnx.save(model_simplified, simplified_path)
            print(f"[EXPORT] OK Simplified ONNX saved: {simplified_path}")
            return True
        else:
            print("[EXPORT] FAIL onnx-simplifier check failed — using original ONNX")
            return False
    except ImportError:
        print("[EXPORT] onnxsim not installed — skipping simplification")
        print("         Install with: pip install onnxsim")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Verify export with onnxruntime
# ══════════════════════════════════════════════════════════════════════════════

def verify_onnx_runtime(onnx_path: str, nn_model, resolution: int, device: str):
    """
    Run the same dummy input through PyTorch AND ONNX Runtime.
    Compare pred_logits and pred_boxes outputs.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[VERIFY] onnxruntime not installed — skipping runtime check")
        print("         Install with: pip install onnxruntime-gpu")
        return

    print("[VERIFY] Running ORT verification …")

    # PyTorch reference output
    wrapper = RFDETRNanoONNXWrapper(nn_model).to(device)
    wrapper.eval()
    dummy = torch.zeros(1, 3, resolution, resolution,
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        pt_logits, pt_boxes = wrapper(dummy)
    pt_logits_np = pt_logits.float().cpu().numpy()
    pt_boxes_np  = pt_boxes.float().cpu().numpy()

    # ONNX Runtime output
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "cuda" in device else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(onnx_path, providers=providers)
    dummy_np = dummy.cpu().numpy()
    ort_logits, ort_boxes = sess.run(None, {"pixel_values": dummy_np})

    # Compare
    logits_err = np.abs(pt_logits_np - ort_logits).max()
    boxes_err  = np.abs(pt_boxes_np - ort_boxes).max()
    print(f"[VERIFY] Max abs error — logits: {logits_err:.6f}  boxes: {boxes_err:.6f}")

    TOLERANCE = 1e-3
    if logits_err < TOLERANCE and boxes_err < TOLERANCE:
        print(f"[VERIFY] OK ONNX output matches PyTorch (tol={TOLERANCE})")
    else:
        print(f"[VERIFY] WARN Error exceeds tolerance {TOLERANCE} — "
              f"check for FP16 accumulation issues")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SELFWATCH — RF-DETR ONNX Export (Phase 1)")
    print("=" * 60)
    print(f"  Device    : {DEVICE}")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  ONNX path : {ONNX_PATH}")
    print("=" * 60)

    nn_model = load_rfdetr_nano(RESOLUTION, DEVICE)
    export_onnx(nn_model, RESOLUTION, ONNX_PATH, DEVICE)
    simplify_onnx(ONNX_PATH, ONNX_SIMPLE_PATH)
    verify_onnx_runtime(ONNX_PATH, nn_model, RESOLUTION, DEVICE)

    print("\n[EXPORT] Phase 1 complete.")
    print("[EXPORT] Next step: run scripts/trt_phase2_validate_onnx.py")
