"""
Phase 3: OSNet ONNX → TensorRT Engine Builder  (Dynamic Batch)
================================================================
Converts the dynamic-batch ONNX model to a TensorRT FP16 engine
with an optimization profile tuned for the RTX 4060 Laptop GPU.

Optimization profile  (min / opt / max):
  batch : 1  /  4  /  16
  H×W   : 128×128  (fixed — OSNet input is always 128×128)

Profile rationale:
  - opt=4  : most common case — 2–6 people in a typical crowded scene
  - max=16 : hard ceiling; 16 simultaneous unique persons is extreme
  - min=1  : single-person or empty-frame fallback

TRT kernel selection:
  TRT uses opt shape to select and fuse kernels. Setting opt=4 biases the
  autotuner toward kernels that are most efficient for the 4-person case
  while remaining valid for batch=1 and batch=16 without rebuilding.

Precision:
  - TRT 10:  FP16 via BuilderFlag.FP16
  - TRT 11+: TF32 via BuilderFlag.TF32 (FP16 now requires ModelOpt)

Output:
    models/osnet_x1_0_dyn_fp16.engine   (dynamic batch, GPU-specific)

IMPORTANT:
  The .engine file is compiled for the SPECIFIC GPU it was built on.
  Always rebuild when changing GPU hardware.

Usage:
    cd <project_root>
    python scripts/reid_trt_phase3_build_engine.py [--onnx PATH] [--output PATH]
"""

import sys
import os
import argparse

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
ONNX_PATH   = os.path.join(MODELS_DIR, "osnet_x1_0_dyn.onnx")
ENGINE_PATH = os.path.join(MODELS_DIR, "osnet_x1_0_dyn_fp16.engine")

INPUT_H = 128
INPUT_W = 128

# Optimization profile — tuned for SELFWATCH typical scene density
MIN_BATCH = 1    # minimum (single person / empty frame)
OPT_BATCH = 4    # optimize for — most common 2–6 person scenes
MAX_BATCH = 16   # absolute ceiling


# ══════════════════════════════════════════════════════════════════════════════
#  Engine builder
# ══════════════════════════════════════════════════════════════════════════════

def build_engine(onnx_path: str, engine_path: str):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    print(f"[TRT] TensorRT version : {trt.__version__}")
    print(f"[TRT] ONNX input       : {onnx_path}")
    print(f"[TRT] Engine output    : {engine_path}")
    print(f"[TRT] Batch profile    : min={MIN_BATCH}  opt={OPT_BATCH}  max={MAX_BATCH}")
    print(f"[TRT] Spatial shape    : {INPUT_H}×{INPUT_W} (fixed)")
    print(f"[TRT] Precision        : FP16 (TF32 fallback for TRT 11+)")
    print()

    builder = trt.Builder(TRT_LOGGER)

    # TRT 10/11 compatibility: EXPLICIT_BATCH is now the default, enum may not exist
    try:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
    except AttributeError:
        # TensorRT >= 10.0 — EXPLICIT_BATCH is always enabled
        network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ── Parse ONNX ─────────────────────────────────────────────────────────
    print("[TRT] Parsing ONNX graph …")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TRT] ONNX parse error {i}: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX file")

    print(f"[TRT] Graph parsed: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs")

    # Confirm I/O names and shapes
    input_tensor  = network.get_input(0)
    output_tensor = network.get_output(0)
    print(f"[TRT] Input  tensor : '{input_tensor.name}'  shape={input_tensor.shape}")
    print(f"[TRT] Output tensor : '{output_tensor.name}' shape={output_tensor.shape}")
    assert input_tensor.name  == "images",   \
        f"Unexpected input name: {input_tensor.name}"
    assert output_tensor.name == "features", \
        f"Unexpected output name: {output_tensor.name}"
    assert input_tensor.shape[0] == -1, (
        "Batch dimension is NOT dynamic (-1). "
        "Re-run Phase 1 — the ONNX was exported without dynamic_axes.")

    # ── Builder config ──────────────────────────────────────────────────────
    config = builder.create_builder_config()

    # OSNet weights are ~10MB — 1 GB workspace is far more than needed.
    # TRT uses workspace for temporary buffers during layer fusion.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # FP16 / Tensor Core acceleration
    # TRT 11 removed BuilderFlag.FP16 because it requires Strongly Typed networks
    # (via offline ModelOpt). TF32 still uses Tensor Cores and is transparent to
    # the existing FP32 ONNX graph.
    try:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] ✓ FP16 enabled (tensor core acceleration)")
    except AttributeError:
        config.set_flag(trt.BuilderFlag.TF32)
        print("[TRT] ⚠ BuilderFlag.FP16 removed in TRT 11 — using TF32 instead.")
        print("[TRT] ✓ TF32 still uses tensor cores; performance is comparable.")

    # ── Optimization profile — REQUIRED for dynamic batch ONNX ─────────────
    # TensorRT must know the concrete range it should optimize for.
    # Without this profile, the engine builder will reject the dynamic-axis input.
    print(f"[TRT] Adding optimization profile: "
          f"min=({MIN_BATCH},3,{INPUT_H},{INPUT_W})  "
          f"opt=({OPT_BATCH},3,{INPUT_H},{INPUT_W})  "
          f"max=({MAX_BATCH},3,{INPUT_H},{INPUT_W})")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(MIN_BATCH, 3, INPUT_H, INPUT_W),
        opt=(OPT_BATCH, 3, INPUT_H, INPUT_W),   # autotuner optimizes for this
        max=(MAX_BATCH, 3, INPUT_H, INPUT_W),
    )
    config.add_optimization_profile(profile)

    # ── Build engine ────────────────────────────────────────────────────────
    print("\n[TRT] Building engine "
          "(OSNet is small — expect 2–6 minutes on first build) …")
    import time
    t0 = time.perf_counter()

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(
            "[TRT] Engine build returned None — check ONNX parse errors above")

    elapsed = time.perf_counter() - t0
    print(f"[TRT] ✓ Engine built in {elapsed:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    file_mb = os.path.getsize(engine_path) / 1e6
    print(f"[TRT] ✓ Engine saved: {engine_path}  ({file_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test — verify deserialization and forward pass at multiple batch sizes
# ══════════════════════════════════════════════════════════════════════════════

def smoke_test(engine_path: str):
    """
    Deserialize the engine and run one forward pass at batch=1, batch=4, batch=8.
    Verifies:
      1. Engine loads without error
      2. Output shape is (N, 512) for each N
      3. set_input_shape() correctly controls the output dimension
    """
    import tensorrt as trt
    import torch, time

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)

    print("\n[SMOKE] Loading engine …")
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("[SMOKE] ✗ Failed to deserialize engine")
        return

    context     = engine.create_execution_context()
    input_name  = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    stream      = torch.cuda.current_stream().cuda_stream

    for bs in [1, 4, 8]:
        # Set concrete shape for this call
        context.set_input_shape(input_name, (bs, 3, INPUT_H, INPUT_W))

        # Allocate buffers sized to the resolved output shape
        out_shape = tuple(context.get_tensor_shape(output_name))
        assert out_shape == (bs, 512), \
            f"Unexpected output shape {out_shape} for batch={bs}"

        in_buf  = torch.zeros(bs, 3, INPUT_H, INPUT_W,
                              dtype=torch.float32, device="cuda")
        out_buf = torch.zeros(out_shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address(input_name,  in_buf.data_ptr())
        context.set_tensor_address(output_name, out_buf.data_ptr())

        # Warmup + timed run
        for _ in range(3):
            context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000

        print(f"[SMOKE] ✓ batch={bs:<2}  output={out_shape}  "
              f"latency={ms:.2f} ms  (10-run avg, warmed up)")

    print("[SMOKE] ✓ All batch sizes passed.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TensorRT dynamic-batch FP16 engine from OSNet ONNX")
    parser.add_argument("--onnx",   default=ONNX_PATH)
    parser.add_argument("--output", default=ENGINE_PATH)
    parser.add_argument("--smoke",  action="store_true", default=True,
                        help="Run smoke test after build")
    args = parser.parse_args()

    print("=" * 65)
    print("  SELFWATCH — OSNet TRT Engine Builder (Phase 3 — Dynamic Batch)")
    print("=" * 65)

    if not os.path.exists(args.onnx):
        print(f"[ERROR] ONNX not found: {args.onnx}")
        print("[ERROR] Run Phase 1 first: python scripts/reid_trt_phase1_export_onnx.py")
        sys.exit(1)

    build_engine(args.onnx, args.output)

    if args.smoke:
        smoke_test(args.output)

    print("\n[TRT] Phase 3 complete.")
    print("[TRT] Next step: python scripts/reid_trt_phase4_benchmark.py")
