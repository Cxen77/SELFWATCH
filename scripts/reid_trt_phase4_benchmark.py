"""
Phase 4: OSNet ReID TensorRT vs PyTorch Benchmark (Dynamic Batch)
===================================================================
Measures exact latency for each stage of the ReID pipeline:
  - PyTorch eager + fast batch preprocessing (current baseline)
  - TensorRT FP16 (Dynamic Batch)

For batch sizes: 1, 4, 8, 16

Measures per-stage using CUDA Events (GPU-accurate):
  - Crop extraction (CPU)
  - Resize (CPU)
  - Batch construction + numpy ops (CPU)
  - CPU → Pinned memory transfer
  - Pinned → GPU transfer (async DMA)
  - GPU normalization
  - Inference (GPU forward)
  - L2 postprocessing
  - Stream sync + GPU → CPU
  - Total pipeline wall time

Expected results on RTX 4060 Laptop:
  PyTorch (batch=8)  : ~30–35 ms inference
  TensorRT (batch=8) : ~6 ms inference     (5x speedup over PyTorch)

Usage:
    cd <project_root>
    python scripts/reid_trt_phase4_benchmark.py
"""

import sys
import os
import time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import cv2

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
ENGINE_PATH = os.path.join(MODELS_DIR, "osnet_x1_0_dyn_fp16.engine")
DEVICE      = "cuda:0"
N_WARMUP    = 10
N_RUNS      = 50

INPUT_H = 128
INPUT_W = 128
_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Max batch size engine was built for
MAX_BATCH = 16


# ══════════════════════════════════════════════════════════════════════════════
#  CUDA Event timer
# ══════════════════════════════════════════════════════════════════════════════

class CudaTimer:
    def __init__(self, stream=None):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)
        self.stream      = stream

    def __enter__(self):
        self.start_event.record(self.stream)
        return self

    def __exit__(self, *_):
        self.end_event.record(self.stream)

    def elapsed_ms(self) -> float:
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic frame and crop generation
# ══════════════════════════════════════════════════════════════════════════════

def make_fake_frame(h: int = 1080, w: int = 1920) -> np.ndarray:
    rng = np.random.RandomState(42)
    return (rng.rand(h, w, 3) * 200).astype(np.uint8)


def extract_boxes(frame: np.ndarray, n: int):
    h, w = frame.shape[:2]
    rng = np.random.RandomState(7)
    boxes = []
    for _ in range(n):
        x1 = rng.randint(0, w - 200)
        y1 = rng.randint(0, h - 300)
        boxes.append((x1, y1, x1 + 128, y1 + 256))
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_results(label: str, times: dict, n_runs: int):
    total = times.get("total", 0)
    print(f"\n  {label}:")
    for k, v in times.items():
        avg = v / n_runs
        print(f"    {k:<24}: {avg:6.3f} ms")
    fps = 1000.0 / max(total / n_runs, 0.001)
    print(f"    {'Throughput':<24}: {fps:.1f} FPS")


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark: PyTorch baseline (uses EmbeddingExtractor._preprocess_batch_fast)
# ══════════════════════════════════════════════════════════════════════════════

def bench_pytorch(extractor, frame: np.ndarray, batch_size: int) -> dict:
    from reid.embedding_extractor import _PIXEL_MEAN, _PIXEL_STD

    boxes = extract_boxes(frame, batch_size)
    stream = extractor._reid_stream

    times = {k: 0.0 for k in [
        "crop", "resize", "batch_construct", "cpu_to_pinned",
        "pinned_to_gpu", "normalization", "inference", "postprocess",
        "sync", "total"
    ]}

    with torch.inference_mode():
        for _ in range(N_WARMUP):
            crops = [cv2.resize(frame[b[1]:b[3], b[0]:b[2]], (INPUT_W, INPUT_H)) for b in boxes]
            extractor.extract_batch(crops)

        for _ in range(N_RUNS):
            t_total = time.perf_counter()

            # 1. Crop
            t0 = time.perf_counter()
            crops_raw = [frame[b[1]:b[3], b[0]:b[2]] for b in boxes]
            times["crop"] += (time.perf_counter() - t0) * 1000

            # 2. Resize
            t0 = time.perf_counter()
            crops = [cv2.resize(c, (INPUT_W, INPUT_H)) for c in crops_raw]
            times["resize"] += (time.perf_counter() - t0) * 1000

            # 3. Batch construct
            t0 = time.perf_counter()
            batch_np = np.stack(crops, axis=0)[:, :, :, ::-1]
            batch_f  = np.ascontiguousarray(
                batch_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            )
            times["batch_construct"] += (time.perf_counter() - t0) * 1000

            # 4. CPU → Pinned
            t0 = time.perf_counter()
            extractor._pinned_buf[:batch_size].copy_(torch.from_numpy(batch_f))
            times["cpu_to_pinned"] += (time.perf_counter() - t0) * 1000

            # 5. Pinned → GPU (async DMA)
            torch.cuda.synchronize()
            with CudaTimer(stream) as ct:
                with torch.cuda.stream(stream):
                    extractor._gpu_buf[:batch_size].copy_(
                        extractor._pinned_buf[:batch_size], non_blocking=True)
            times["pinned_to_gpu"] += ct.elapsed_ms()

            # 6. GPU Normalization
            with CudaTimer(stream) as ct:
                with torch.cuda.stream(stream):
                    gpu_slice = extractor._gpu_buf[:batch_size]
                    gpu_slice = (gpu_slice - extractor._gpu_mean) / extractor._gpu_std
            times["normalization"] += ct.elapsed_ms()

            # 7. Inference
            with CudaTimer(stream) as ct:
                with torch.cuda.stream(stream):
                    with torch.autocast(device_type="cuda", enabled=extractor._half):
                        feats = extractor._model(gpu_slice)
            times["inference"] += ct.elapsed_ms()

            # 8. Postprocess (L2 norm)
            with CudaTimer(stream) as ct:
                with torch.cuda.stream(stream):
                    feats = F.normalize(feats, p=2, dim=1)
            times["postprocess"] += ct.elapsed_ms()

            # 9. Sync + GPU → CPU
            t0 = time.perf_counter()
            stream.synchronize()
            _ = feats.detach().float().cpu().numpy()
            times["sync"] += (time.perf_counter() - t0) * 1000

            times["total"] += (time.perf_counter() - t_total) * 1000

    return times


# ══════════════════════════════════════════════════════════════════════════════
#  Load TRT engine
# ══════════════════════════════════════════════════════════════════════════════

def load_trt_engine(engine_path: str):
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark: TensorRT (Dynamic Batch)
# ══════════════════════════════════════════════════════════════════════════════

def bench_trt(engine, frame: np.ndarray, batch_size: int) -> dict:
    """
    Benchmarks TRT ReID pipeline with DYNAMIC batching.
    Performs exactly ONE engine forward pass for N crops.
    """
    import tensorrt as trt

    context     = engine.create_execution_context()
    input_name  = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    trt_stream  = torch.cuda.Stream(device=DEVICE)

    # Set shape so buffers allocate correctly
    context.set_input_shape(input_name, (MAX_BATCH, 3, INPUT_H, INPUT_W))

    # Pre-allocated max buffers
    input_buf  = torch.zeros(MAX_BATCH, 3, INPUT_H, INPUT_W, dtype=torch.float32, device=DEVICE)
    output_buf = torch.zeros(MAX_BATCH, 512, dtype=torch.float32, device=DEVICE)
    context.set_tensor_address(input_name,  input_buf.data_ptr())
    context.set_tensor_address(output_name, output_buf.data_ptr())

    # Pre-allocated pinned memory
    pinned_buf = torch.zeros(MAX_BATCH, 3, INPUT_H, INPUT_W, dtype=torch.float32, pin_memory=True)

    _mean_gpu = torch.tensor(_MEAN, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    _std_gpu  = torch.tensor(_STD,  dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)

    boxes = extract_boxes(frame, batch_size)

    times = {k: 0.0 for k in [
        "crop", "resize", "batch_construct", "cpu_to_pinned",
        "pinned_to_gpu", "normalization", "inference", "postprocess",
        "sync", "total"
    ]}

    with torch.inference_mode():
        # Warmup
        for _ in range(N_WARMUP):
            crops = [cv2.resize(frame[b[1]:b[3], b[0]:b[2]], (INPUT_W, INPUT_H)) for b in boxes]
            batch_np = np.stack(crops, axis=0)[:, :, :, ::-1]
            batch_f  = np.ascontiguousarray(batch_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
            
            pinned_buf[:batch_size].copy_(torch.from_numpy(batch_f))
            with torch.cuda.stream(trt_stream):
                input_buf[:batch_size].copy_(pinned_buf[:batch_size], non_blocking=True)
                gpu_slice = input_buf[:batch_size]
                gpu_slice.sub_(_mean_gpu).div_(_std_gpu)
                
                context.set_input_shape(input_name, (batch_size, 3, INPUT_H, INPUT_W))
                context.execute_async_v3(stream_handle=trt_stream.cuda_stream)
                emb = F.normalize(output_buf[:batch_size], p=2, dim=1)
            trt_stream.synchronize()

        # Measurement
        for _ in range(N_RUNS):
            t_total = time.perf_counter()

            # 1. Crop
            t0 = time.perf_counter()
            crops_raw = [frame[b[1]:b[3], b[0]:b[2]] for b in boxes]
            times["crop"] += (time.perf_counter() - t0) * 1000

            # 2. Resize
            t0 = time.perf_counter()
            crops = [cv2.resize(c, (INPUT_W, INPUT_H)) for c in crops_raw]
            times["resize"] += (time.perf_counter() - t0) * 1000

            # 3. Batch construct
            t0 = time.perf_counter()
            batch_np = np.stack(crops, axis=0)[:, :, :, ::-1]
            batch_f  = np.ascontiguousarray(
                batch_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            )
            times["batch_construct"] += (time.perf_counter() - t0) * 1000

            # 4. CPU → Pinned
            t0 = time.perf_counter()
            pinned_buf[:batch_size].copy_(torch.from_numpy(batch_f))
            times["cpu_to_pinned"] += (time.perf_counter() - t0) * 1000

            # 5. Pinned → GPU
            torch.cuda.synchronize()
            with CudaTimer(trt_stream) as ct:
                with torch.cuda.stream(trt_stream):
                    gpu_slice = input_buf[:batch_size]
                    gpu_slice.copy_(pinned_buf[:batch_size], non_blocking=True)
            times["pinned_to_gpu"] += ct.elapsed_ms()

            # 6. GPU Normalization
            with CudaTimer(trt_stream) as ct:
                with torch.cuda.stream(trt_stream):
                    gpu_slice.sub_(_mean_gpu).div_(_std_gpu)
            times["normalization"] += ct.elapsed_ms()

            # 7. TRT Inference (Dynamic batch — 1 call for N crops)
            with CudaTimer(trt_stream) as ct:
                with torch.cuda.stream(trt_stream):
                    context.set_input_shape(input_name, (batch_size, 3, INPUT_H, INPUT_W))
                    context.execute_async_v3(stream_handle=trt_stream.cuda_stream)
            times["inference"] += ct.elapsed_ms()

            # 8. L2 norm postprocess
            with CudaTimer(trt_stream) as ct:
                with torch.cuda.stream(trt_stream):
                    emb = F.normalize(output_buf[:batch_size], p=2, dim=1)
            times["postprocess"] += ct.elapsed_ms()

            # 9. Sync + GPU → CPU
            t0 = time.perf_counter()
            trt_stream.synchronize()
            _ = emb.detach().float().cpu().numpy()
            times["sync"] += (time.perf_counter() - t0) * 1000

            times["total"] += (time.perf_counter() - t_total) * 1000

    return times


# ══════════════════════════════════════════════════════════════════════════════
#  Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: dict):
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY — OSNet ReID TensorRT vs PyTorch (Dynamic Batch)")
    print("=" * 70)
    print(f"  {'Backend':<32} {'Inference':<14} {'Total':<14} {'FPS'}")
    print("  " + "-" * 66)
    for name, (times, n_runs) in all_results.items():
        inf_ms   = times["inference"] / n_runs
        total_ms = times["total"]     / n_runs
        fps      = 1000.0 / max(total_ms, 0.001)
        print(f"  {name:<32} {inf_ms:>8.2f} ms   {total_ms:>8.2f} ms   {fps:.1f}")
    print("=" * 70)

    # Speedup table
    for bs in [1, 4, 8]:
        pt = all_results.get(f"PyTorch (batch={bs})")
        tr = all_results.get(f"TRT (batch={bs})")
        if pt and tr:
            pt_inf = pt[0]["inference"] / N_RUNS
            trt_inf = tr[0]["inference"] / N_RUNS
            print(f"\n  TRT vs PyTorch (batch={bs}):")
            print(f"    Inference speedup : {pt_inf / max(trt_inf, 0.001):.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  SELFWATCH — OSNet ReID TRT vs PyTorch Benchmark (Phase 4)")
    print("=" * 65)

    from reid.embedding_extractor import EmbeddingExtractor
    frame = make_fake_frame()
    all_results = {}

    # ── PyTorch Baseline ────────────────────────────────────────────────────
    print("\n[BENCH] Loading PyTorch OSNet (EmbeddingExtractor) …")
    extractor = EmbeddingExtractor(half=True)

    for bs in [1, 4, 8, 16]:
        label = f"PyTorch (batch={bs})"
        print(f"\n[BENCH] {label} …")
        times = bench_pytorch(extractor, frame, bs)
        _print_results(label, times, N_RUNS)
        all_results[label] = (times, N_RUNS)

    del extractor
    torch.cuda.empty_cache()

    # ── TensorRT ────────────────────────────────────────────────────────────
    if os.path.exists(ENGINE_PATH):
        print("\n[BENCH] Loading TRT engine …")
        try:
            import tensorrt as trt
            print(f"[BENCH] TensorRT version: {trt.__version__}")
            engine = load_trt_engine(ENGINE_PATH)

            for bs in [1, 4, 8, 16]:
                label = f"TRT (batch={bs})"
                print(f"\n[BENCH] {label} …")
                times = bench_trt(engine, frame, bs)
                _print_results(label, times, N_RUNS)
                all_results[label] = (times, N_RUNS)

            del engine
        except Exception as e:
            print(f"[BENCH] TRT skipped: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"[BENCH] Engine not found — skipping TRT: {ENGINE_PATH}")
        print("[BENCH] Run Phase 3 first: python scripts/reid_trt_phase3_build_engine.py")

    print_summary(all_results)
