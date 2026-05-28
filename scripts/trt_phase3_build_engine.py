"""
Phase 3: ONNX → TensorRT Engine Builder
=========================================
Converts the validated ONNX model to a TensorRT FP16 engine
optimized for the RTX 4060 Laptop GPU.

Engine configuration:
  - FP16 mode (tensor core acceleration)
  - Dynamic batch: min=1, opt=2, max=8
    (opt=2 matches typical 2-camera use-case)
  - Dynamic H/W: fixed at RESOLUTION=384 (no dynamic spatial dims)
  - Workspace: 2 GB
  - Precision: FP16 primary, FP32 accumulation where needed

Output:
    models/rfdetr_nano_384_fp16.engine   (TRT engine, GPU-specific)

IMPORTANT:
  The .engine file is compiled for the SPECIFIC GPU it was built on.
  It will NOT transfer to a different GPU (different VRAM, compute cap).
  Always rebuild when changing GPU.

Usage:
    python scripts/trt_phase3_build_engine.py [--onnx PATH] [--output PATH]
"""

import sys
import os
import argparse

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
ONNX_PATH   = os.path.join(MODELS_DIR, "rfdetr_nano_384.onnx")
ENGINE_PATH = os.path.join(MODELS_DIR, "rfdetr_nano_384_fp16.engine")

RESOLUTION  = 384
MIN_BATCH   = 1
OPT_BATCH   = 2    # Most common: 2-camera batch
MAX_BATCH   = 8    # Safety margin


def build_engine(onnx_path: str, engine_path: str):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    print(f"[TRT] TensorRT version : {trt.__version__}")
    print(f"[TRT] ONNX input       : {onnx_path}")
    print(f"[TRT] Engine output    : {engine_path}")
    print(f"[TRT] Batch profile    : min={MIN_BATCH} opt={OPT_BATCH} max={MAX_BATCH}")
    print(f"[TRT] Precision        : FP16")
    print()

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    # ── Parse ONNX ─────────────────────────────────────────────────────────
    print("[TRT] Parsing ONNX graph …")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TRT] ONNX parse error {i}: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX file")
    print(f"[TRT] Graph parsed: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs")

    # ── Builder config ──────────────────────────────────────────────────────
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

    # FP16 mode — mandatory for tensor core acceleration on RTX 4060
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] ✓ FP16 enabled (tensor core acceleration)")
    else:
        print("[TRT] ⚠ FP16 not available — building FP32 engine")

    # ── Dynamic batch profile ───────────────────────────────────────────────
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    print(f"[TRT] Input tensor name: '{input_name}'")

    profile.set_shape(
        input_name,
        min=(MIN_BATCH, 3, RESOLUTION, RESOLUTION),
        opt=(OPT_BATCH, 3, RESOLUTION, RESOLUTION),
        max=(MAX_BATCH, 3, RESOLUTION, RESOLUTION),
    )
    config.add_optimization_profile(profile)

    # ── Build engine ────────────────────────────────────────────────────────
    print("[TRT] Building engine (this takes 2–10 minutes, normal for first build) …")
    import time
    t0 = time.perf_counter()

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("[TRT] Engine build failed — check ONNX parse errors above")

    elapsed = time.perf_counter() - t0
    print(f"[TRT] ✓ Engine built in {elapsed:.1f}s")

    # ── Save engine ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    file_mb = os.path.getsize(engine_path) / 1e6
    print(f"[TRT] ✓ Engine saved: {engine_path}  ({file_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
#  Quick engine smoke-test (verifies deserialization and a single forward)
# ══════════════════════════════════════════════════════════════════════════════

def smoke_test(engine_path: str):
    import tensorrt as trt
    import numpy as np, torch, time

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)

    print("\n[SMOKE] Loading engine for smoke test …")
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("[SMOKE] ✗ Failed to deserialize engine")
        return

    context = engine.create_execution_context()

    # Set batch=1 input shape
    input_name = engine.get_tensor_name(0)
    context.set_input_shape(input_name, (1, 3, RESOLUTION, RESOLUTION))

    # Allocate IO buffers
    import ctypes
    bindings = []
    device_mems = []
    output_tensors = []

    for i in range(engine.num_io_tensors):
        name   = engine.get_tensor_name(i)
        mode   = engine.get_tensor_mode(name)
        shape  = context.get_tensor_shape(name)
        dtype  = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        d_mem = torch.zeros(int(np.prod(shape)), dtype=torch.float32,
                            device="cuda").contiguous()
        device_mems.append(d_mem)
        context.set_tensor_address(name, d_mem.data_ptr())

        if mode == trt.TensorIOMode.OUTPUT:
            output_tensors.append((name, shape, d_mem))

    # Dummy input
    dummy = torch.zeros(1, 3, RESOLUTION, RESOLUTION,
                        dtype=torch.float32, device="cuda")
    context.set_tensor_address(input_name, dummy.data_ptr())

    # Forward
    stream = torch.cuda.current_stream().cuda_stream
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000

    print(f"[SMOKE] ✓ Engine forward pass: {ms:.1f} ms  (first-pass — warmup needed)")
    for name, shape, buf in output_tensors:
        print(f"[SMOKE]   output '{name}': shape={tuple(shape)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT FP16 engine from ONNX")
    parser.add_argument("--onnx",   default=ONNX_PATH)
    parser.add_argument("--output", default=ENGINE_PATH)
    parser.add_argument("--smoke",  action="store_true", default=True,
                        help="Run smoke test after build")
    args = parser.parse_args()

    print("=" * 60)
    print("  SELFWATCH — TensorRT Engine Builder (Phase 3)")
    print("=" * 60)

    if not os.path.exists(args.onnx):
        print(f"[ERROR] ONNX not found: {args.onnx}")
        print("[ERROR] Run Phase 1 first: python scripts/trt_phase1_export_onnx.py")
        sys.exit(1)

    build_engine(args.onnx, args.output)

    if args.smoke:
        smoke_test(args.output)

    print("\n[TRT] Phase 3 complete.")
    print("[TRT] Next step: python scripts/trt_phase4_benchmark.py")
