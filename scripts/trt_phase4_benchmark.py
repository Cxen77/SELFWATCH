"""
Phase 4: TensorRT vs PyTorch Benchmark
========================================
Measures exact latency for:
  - PyTorch eager (current baseline)
  - ONNX Runtime (CUDA EP)
  - TensorRT FP16

For both batch=1 (single camera) and batch=2 (two-camera batching).

Measures:
  - preprocessing time (CPU)
  - GPU transfer time (CPU→GPU)
  - inference time (GPU forward)
  - postprocessing time (GPU decode)
  - total latency

Expected results on RTX 4060 Laptop:
  PyTorch eager  : ~65–85 ms/frame
  ONNX Runtime   : ~30–50 ms/frame
  TensorRT FP16  : ~12–25 ms/frame

Usage:
    python scripts/trt_phase4_benchmark.py
"""

import sys
import os
import time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import cv2

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
ONNX_PATH   = os.path.join(MODELS_DIR, "rfdetr_nano_384.onnx")
ENGINE_PATH = os.path.join(MODELS_DIR, "rfdetr_nano_384_fp16.engine")
RESOLUTION  = 384
DEVICE      = "cuda:0"
N_RUNS      = 50
N_WARMUP    = 10

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MEAN_GPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
_STD_GPU  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_fake_frames(n: int, h: int = 540, w: int = 960) -> list:
    rng = np.random.RandomState(42)
    return [(rng.rand(h, w, 3) * 200).astype(np.uint8) for _ in range(n)]


def preprocess_pytorch_batch(frames: list) -> torch.Tensor:
    """Mirror of rtdetr_detector._preprocess_frame — GPU path."""
    import torch.nn.functional as Fnn
    tensors = []
    for frame in frames:
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cpu_t = torch.from_numpy(rgb)
        gpu_u = cpu_t.to(DEVICE, non_blocking=True)
        gpu_f = gpu_u.permute(2, 0, 1).float().mul_(1.0 / 255.0)
        gpu_f = Fnn.interpolate(
            gpu_f.unsqueeze(0), size=(RESOLUTION, RESOLUTION),
            mode="bilinear", align_corners=False, antialias=True)
        gpu_f = (gpu_f - _MEAN_GPU) / _STD_GPU
        tensors.append(gpu_f)
    return torch.cat(tensors, dim=0)


def preprocess_numpy_batch(frames: list) -> np.ndarray:
    """CPU numpy preprocessing for ORT/TRT path."""
    batches = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (RESOLUTION, RESOLUTION)).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        batches.append(rgb.transpose(2, 0, 1))
    return np.stack(batches, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  CUDA event timer helper
# ══════════════════════════════════════════════════════════════════════════════

class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *_):
        self.end_event.record()

    def elapsed_ms(self) -> float:
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark: PyTorch eager
# ══════════════════════════════════════════════════════════════════════════════

def bench_pytorch(nn_model, frames: list, label: str):
    print(f"\n[BENCH] PyTorch Eager — {label}")

    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            batch = preprocess_pytorch_batch(frames)
            _ = nn_model(batch)
    torch.cuda.synchronize()

    preproc_times, infer_times, total_times = [], [], []
    for _ in range(N_RUNS):
        t_total0 = time.perf_counter()

        with CudaTimer() as ct_pre:
            batch = preprocess_pytorch_batch(frames)
        torch.cuda.synchronize()
        preproc_ms = ct_pre.elapsed_ms()

        with CudaTimer() as ct_infer:
            with torch.no_grad():
                _ = nn_model(batch)
        torch.cuda.synchronize()
        infer_ms = ct_infer.elapsed_ms()

        total_ms = (time.perf_counter() - t_total0) * 1000
        preproc_times.append(preproc_ms)
        infer_times.append(infer_ms)
        total_times.append(total_ms)

    print(f"  Preprocess  : {np.mean(preproc_times):.1f} ± {np.std(preproc_times):.1f} ms")
    print(f"  Inference   : {np.mean(infer_times):.1f} ± {np.std(infer_times):.1f} ms")
    print(f"  Total wall  : {np.mean(total_times):.1f} ± {np.std(total_times):.1f} ms")
    print(f"  Throughput  : {1000.0 / np.mean(total_times):.1f} FPS")
    return {
        "preprocess_ms": float(np.mean(preproc_times)),
        "inference_ms":  float(np.mean(infer_times)),
        "total_ms":      float(np.mean(total_times)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark: ONNX Runtime
# ══════════════════════════════════════════════════════════════════════════════

def bench_ort(sess, frames: list, label: str):
    print(f"\n[BENCH] ONNX Runtime — {label}")

    batch_np = preprocess_numpy_batch(frames)

    for _ in range(N_WARMUP):
        sess.run(None, {"pixel_values": batch_np})

    preproc_times, infer_times, total_times = [], [], []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        batch_np = preprocess_numpy_batch(frames)
        preproc_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        sess.run(None, {"pixel_values": batch_np})
        infer_ms = (time.perf_counter() - t1) * 1000

        preproc_times.append(preproc_ms)
        infer_times.append(infer_ms)
        total_times.append(preproc_ms + infer_ms)

    print(f"  Preprocess  : {np.mean(preproc_times):.1f} ± {np.std(preproc_times):.1f} ms")
    print(f"  Inference   : {np.mean(infer_times):.1f} ± {np.std(infer_times):.1f} ms")
    print(f"  Total wall  : {np.mean(total_times):.1f} ± {np.std(total_times):.1f} ms")
    print(f"  Throughput  : {1000.0 / np.mean(total_times):.1f} FPS")
    return {
        "preprocess_ms": float(np.mean(preproc_times)),
        "inference_ms":  float(np.mean(infer_times)),
        "total_ms":      float(np.mean(total_times)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark: TensorRT
# ══════════════════════════════════════════════════════════════════════════════

def load_trt_engine(engine_path: str):
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def bench_trt(engine, frames: list, label: str):
    import tensorrt as trt
    print(f"\n[BENCH] TensorRT FP16 — {label}")

    n = len(frames)
    context = engine.create_execution_context()

    input_name  = engine.get_tensor_name(0)
    output_names = [engine.get_tensor_name(i)
                    for i in range(1, engine.num_io_tensors)]

    # Set dynamic input shape for this batch size
    context.set_input_shape(input_name, (n, 3, RESOLUTION, RESOLUTION))

    # Allocate output buffers
    out_bufs = {}
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        buf   = torch.zeros(shape, dtype=torch.float32, device=DEVICE)
        out_bufs[name] = buf
        context.set_tensor_address(name, buf.data_ptr())

    # Pinned input buffer
    in_buf = torch.zeros(n, 3, RESOLUTION, RESOLUTION,
                         dtype=torch.float32, device=DEVICE)
    context.set_tensor_address(input_name, in_buf.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream

    def run_once(frames):
        batch_np = preprocess_numpy_batch(frames)
        in_buf.copy_(torch.from_numpy(batch_np).to(DEVICE, non_blocking=True))
        context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(N_WARMUP):
        run_once(frames)

    preproc_times, infer_times, total_times = [], [], []
    for _ in range(N_RUNS):
        t_total0 = time.perf_counter()

        t0 = time.perf_counter()
        batch_np = preprocess_numpy_batch(frames)
        in_tensor = torch.from_numpy(batch_np).to(DEVICE, non_blocking=True)
        preproc_ms = (time.perf_counter() - t0) * 1000

        with CudaTimer() as ct:
            in_buf.copy_(in_tensor)
            context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        infer_ms = ct.elapsed_ms()

        total_ms = (time.perf_counter() - t_total0) * 1000
        preproc_times.append(preproc_ms)
        infer_times.append(infer_ms)
        total_times.append(total_ms)

    print(f"  Preprocess  : {np.mean(preproc_times):.1f} ± {np.std(preproc_times):.1f} ms")
    print(f"  Inference   : {np.mean(infer_times):.1f} ± {np.std(infer_times):.1f} ms")
    print(f"  Total wall  : {np.mean(total_times):.1f} ± {np.std(total_times):.1f} ms")
    print(f"  Throughput  : {1000.0 / np.mean(total_times):.1f} FPS")
    return {
        "preprocess_ms": float(np.mean(preproc_times)),
        "inference_ms":  float(np.mean(infer_times)),
        "total_ms":      float(np.mean(total_times)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VRAM measurement
# ══════════════════════════════════════════════════════════════════════════════

def report_vram():
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved  = torch.cuda.memory_reserved() / 1e6
    print(f"\n[VRAM] Allocated: {allocated:.0f} MB  Reserved: {reserved:.0f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  {'Backend':<22} {'Preprocess':>12} {'Inference':>12} {'Total':>12}")
    print("  " + "-" * 60)
    for name, r in results.items():
        print(f"  {name:<22} {r['preprocess_ms']:>10.1f}ms "
              f"{r['inference_ms']:>10.1f}ms "
              f"{r['total_ms']:>10.1f}ms")
    print("=" * 65)

    # Speedup vs PyTorch
    if "PyTorch (batch=1)" in results and "TensorRT (batch=1)" in results:
        pt  = results["PyTorch (batch=1)"]["total_ms"]
        trt = results["TensorRT (batch=1)"]["total_ms"]
        print(f"\n  TensorRT speedup vs PyTorch (batch=1): {pt/trt:.2f}x")

    if "PyTorch (batch=2)" in results and "TensorRT (batch=2)" in results:
        pt  = results["PyTorch (batch=2)"]["total_ms"]
        trt = results["TensorRT (batch=2)"]["total_ms"]
        print(f"  TensorRT speedup vs PyTorch (batch=2): {pt/trt:.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  SELFWATCH — TensorRT vs PyTorch Benchmark (Phase 4)")
    print("=" * 65)

    results = {}
    frames_1 = make_fake_frames(1)
    frames_2 = make_fake_frames(2)

    # ── PyTorch ────────────────────────────────────────────────────────────
    print("\n[BENCH] Loading PyTorch model …")
    from rfdetr import RFDETRNano
    rfdetr = RFDETRNano(resolution=RESOLUTION)
    nn_model = rfdetr.model.model.eval().to(DEVICE)

    results["PyTorch (batch=1)"] = bench_pytorch(nn_model, frames_1, "batch=1")
    results["PyTorch (batch=2)"] = bench_pytorch(nn_model, frames_2, "batch=2")

    del rfdetr, nn_model
    torch.cuda.empty_cache()

    # ── ONNX Runtime ───────────────────────────────────────────────────────
    if os.path.exists(ONNX_PATH):
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(ONNX_PATH, providers=providers)
            results["ORT CUDA (batch=1)"] = bench_ort(sess, frames_1, "batch=1")
            results["ORT CUDA (batch=2)"] = bench_ort(sess, frames_2, "batch=2")
            del sess
        except Exception as e:
            print(f"[BENCH] ORT skipped: {e}")
    else:
        print(f"[BENCH] ONNX not found, skipping ORT benchmark: {ONNX_PATH}")

    # ── TensorRT ───────────────────────────────────────────────────────────
    if os.path.exists(ENGINE_PATH):
        try:
            engine = load_trt_engine(ENGINE_PATH)
            results["TensorRT (batch=1)"] = bench_trt(engine, frames_1, "batch=1")
            results["TensorRT (batch=2)"] = bench_trt(engine, frames_2, "batch=2")
            del engine
        except Exception as e:
            print(f"[BENCH] TensorRT skipped: {e}")
    else:
        print(f"[BENCH] Engine not found, skipping TRT benchmark: {ENGINE_PATH}")

    report_vram()
    print_summary(results)

    # Save results
    import json
    out_path = os.path.join(MODELS_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[BENCH] Results saved: {out_path}")
    print("[BENCH] Phase 4 complete. Review results before Phase 5 integration.")
