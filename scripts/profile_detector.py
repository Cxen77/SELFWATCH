"""
Detection Pipeline Profiler — Before/After Comparison
======================================================
Measures the old PIL path vs. the new optimized (no-PIL) path.

Run:
    python scripts/profile_detector.py
"""

import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn
import torchvision.transforms.functional as F
from PIL import Image

WARMUP_RUNS = 6
PROFILE_RUNS = 30
RESOLUTION = 384
FRAME_H, FRAME_W = 1080, 1920

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def ts():
    sync()
    return time.perf_counter()

def ms(start, end):
    return (end - start) * 1000.0

def stat(vals):
    arr = np.array(vals)
    return f"avg={arr.mean():.2f}ms  min={arr.min():.2f}ms  p95={np.percentile(arr,95):.2f}ms"

# ── Load via our optimized RTDETRDetector ────────────────────────────────────
print("Loading RT-DETR Nano via RTDETRDetector (optimized path)...")
t0 = time.perf_counter()
from detectors.rtdetr_detector import RTDETRDetector
detector = RTDETRDetector(variant="nano", resolution=RESOLUTION,
                          use_amp=True, compile_model=True)
print(f"Loaded in {(time.perf_counter()-t0)*1000:.0f}ms\n")

device_str = detector._device
device      = torch.device(device_str)
model_wrapper = detector._model
internal    = model_wrapper.model
inf_dtype   = detector._inf_dtype
is_opt      = getattr(model_wrapper, '_is_optimized_for_inference', False)
means_gpu   = detector._mean_gpu
stds_gpu    = detector._std_gpu

print(f"Device:    {device}")
print(f"Optimized: {is_opt}")
print(f"dtype:     {inf_dtype}")
print(f"GPU:       {torch.cuda.get_device_name(0)}\n")

dummy_frame = np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)

# ── Warmup ───────────────────────────────────────────────────────────────────
print(f"Warming up ({WARMUP_RUNS} runs)...")
for _ in range(WARMUP_RUNS):
    detector.detect(dummy_frame, conf_threshold=0.35, target_classes=[0])
sync()
print("Warmup done.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK A: OLD PATH (PIL based — what model_wrapper.predict() does)
# ─────────────────────────────────────────────────────────────────────────────
pil_timings = {k: [] for k in [
    "cvt_color", "pil_from_array", "to_tensor", "cpu_to_gpu",
    "resize_gpu", "normalize_gpu", "forward", "postprocess", "gpu_to_cpu", "total"
]}

print(f"Benchmarking OLD path (PIL)  — {PROFILE_RUNS} runs...")
target_sizes = torch.tensor([[FRAME_H, FRAME_W]], device=device)

for _ in range(PROFILE_RUNS):
    t_total0 = ts()

    t = ts(); rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB); t1 = ts()
    pil_timings["cvt_color"].append(ms(t, t1))

    t = ts(); pil_img = Image.fromarray(rgb); t1 = ts()
    pil_timings["pil_from_array"].append(ms(t, t1))

    t = ts(); img_tensor = F.to_tensor(pil_img); t1 = ts()
    pil_timings["to_tensor"].append(ms(t, t1))

    t = ts(); img_gpu = img_tensor.to(device); sync(); t1 = ts()
    pil_timings["cpu_to_gpu"].append(ms(t, t1))

    t = ts(); img_gpu = F.resize(img_gpu, [RESOLUTION, RESOLUTION]); sync(); t1 = ts()
    pil_timings["resize_gpu"].append(ms(t, t1))

    means = getattr(model_wrapper, 'means', [0.485, 0.456, 0.406])
    stds  = getattr(model_wrapper, 'stds',  [0.229, 0.224, 0.225])
    t = ts(); img_gpu = F.normalize(img_gpu, means, stds); sync(); t1 = ts()
    pil_timings["normalize_gpu"].append(ms(t, t1))

    batch = img_gpu.unsqueeze(0)

    t = ts()
    with torch.no_grad():
        if is_opt and internal.inference_model is not None:
            predictions = internal.inference_model(batch.to(dtype=inf_dtype))
        else:
            predictions = internal.model(batch)
        if isinstance(predictions, tuple):
            predictions = {"pred_logits": predictions[1], "pred_boxes": predictions[0]}
    sync(); t1 = ts()
    pil_timings["forward"].append(ms(t, t1))

    t = ts()
    with torch.no_grad():
        results = internal.postprocess(predictions, target_sizes=target_sizes)
    sync(); t1 = ts()
    pil_timings["postprocess"].append(ms(t, t1))

    result = results[0]
    keep = result["scores"] > 0.35
    t = ts()
    _ = result["boxes"][keep].float().cpu().numpy()
    _ = result["scores"][keep].float().cpu().numpy()
    _ = result["labels"][keep].cpu().numpy()
    sync(); t1 = ts()
    pil_timings["gpu_to_cpu"].append(ms(t, t1))

    pil_timings["total"].append(ms(t_total0, ts()))

# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK B: NEW PATH (no-PIL, GPU normalize + interpolate)
# ─────────────────────────────────────────────────────────────────────────────
new_timings = {k: [] for k in [
    "cvt_color", "numpy_to_gpu", "hwc_to_chw_norm", "resize_gpu",
    "forward", "postprocess", "gpu_to_cpu", "total"
]}

print(f"Benchmarking NEW path (no PIL) — {PROFILE_RUNS} runs...")

for _ in range(PROFILE_RUNS):
    t_total0 = ts()

    t = ts(); rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB); t1 = ts()
    new_timings["cvt_color"].append(ms(t, t1))

    t = ts()
    cpu_t  = torch.from_numpy(rgb)
    gpu_u8 = cpu_t.to(device, non_blocking=True)
    sync(); t1 = ts()
    new_timings["numpy_to_gpu"].append(ms(t, t1))

    t = ts()
    gpu_f = gpu_u8.permute(2, 0, 1).float().mul_(1.0 / 255.0)
    gpu_f = (gpu_f - means_gpu) / stds_gpu
    sync(); t1 = ts()
    new_timings["hwc_to_chw_norm"].append(ms(t, t1))

    t = ts()
    batch = Fnn.interpolate(
        gpu_f.unsqueeze(0), size=(RESOLUTION, RESOLUTION),
        mode='bilinear', align_corners=False)
    sync(); t1 = ts()
    new_timings["resize_gpu"].append(ms(t, t1))

    t = ts()
    with torch.no_grad():
        if is_opt and internal.inference_model is not None:
            predictions = internal.inference_model(batch.to(dtype=inf_dtype))
        else:
            predictions = internal.model(batch)
        if isinstance(predictions, tuple):
            predictions = {"pred_logits": predictions[1], "pred_boxes": predictions[0]}
    sync(); t1 = ts()
    new_timings["forward"].append(ms(t, t1))

    t = ts()
    with torch.no_grad():
        results = internal.postprocess(predictions, target_sizes=target_sizes)
    sync(); t1 = ts()
    new_timings["postprocess"].append(ms(t, t1))

    result = results[0]
    keep = result["scores"] > 0.35
    t = ts()
    _ = result["boxes"][keep].float().cpu().numpy()
    _ = result["scores"][keep].float().cpu().numpy()
    _ = result["labels"][keep].cpu().numpy()
    sync(); t1 = ts()
    new_timings["gpu_to_cpu"].append(ms(t, t1))

    new_timings["total"].append(ms(t_total0, ts()))

# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK C: Our RTDETRDetector.detect() wall time
# ─────────────────────────────────────────────────────────────────────────────
wall_times = []
print(f"Benchmarking RTDETRDetector.detect() end-to-end — {PROFILE_RUNS} runs...")
for _ in range(PROFILE_RUNS):
    t = ts()
    detector.detect(dummy_frame, conf_threshold=0.35, target_classes=[0])
    wall_times.append(ms(t, ts()))

# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"  RF-DETR Nano  |  {RESOLUTION}x{RESOLUTION}  |  {FRAME_W}x{FRAME_H} input")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  FP16+compile: {is_opt}")
print("=" * 72)

print("\n  OLD PATH (PIL-based):")
old_stages = [
    ("  1. cv2.cvtColor [CPU]",        "cvt_color"),
    ("  2. PIL.fromarray [CPU]",        "pil_from_array"),
    ("  3. F.to_tensor [CPU]",         "to_tensor"),
    ("  4. CPU->GPU float32 [PCIe]",   "cpu_to_gpu"),
    ("  5. F.resize on GPU",           "resize_gpu"),
    ("  6. F.normalize on GPU",        "normalize_gpu"),
    ("  7. Model forward [GPU]",       "forward"),
    ("  8. Postprocess [GPU]",         "postprocess"),
    ("  9. GPU->CPU numpy [PCIe]",     "gpu_to_cpu"),
]
old_total = 0
for label, key in old_stages:
    avg = np.mean(pil_timings[key])
    old_total += avg
    print(f"  {label:<40}  {stat(pil_timings[key])}")
print(f"  {'TOTAL ACCOUNTED':<40}  avg={old_total:.2f}ms")
print(f"  {'TOTAL WALL (our measure)':<40}  {stat(pil_timings['total'])}")

print("\n  NEW PATH (no-PIL, GPU preprocess):")
new_stages = [
    ("  1. cv2.cvtColor [CPU]",        "cvt_color"),
    ("  2. numpy->GPU uint8 [PCIe]",   "numpy_to_gpu"),
    ("  3. HWC->CHW + normalize [GPU]","hwc_to_chw_norm"),
    ("  4. interpolate resize [GPU]",  "resize_gpu"),
    ("  5. Model forward [GPU]",       "forward"),
    ("  6. Postprocess [GPU]",         "postprocess"),
    ("  7. GPU->CPU numpy [PCIe]",     "gpu_to_cpu"),
]
new_total = 0
for label, key in new_stages:
    avg = np.mean(new_timings[key])
    new_total += avg
    print(f"  {label:<40}  {stat(new_timings[key])}")
print(f"  {'TOTAL ACCOUNTED':<40}  avg={new_total:.2f}ms")
print(f"  {'TOTAL WALL (our measure)':<40}  {stat(new_timings['total'])}")

print(f"\n  RTDETRDetector.detect() end-to-end:  {stat(wall_times)}")

old_avg = np.mean(pil_timings["total"])
new_avg = new_total
wall_avg = np.mean(wall_times)
speedup = old_avg / wall_avg if wall_avg > 0 else 0
saved = old_avg - wall_avg
print(f"\n  SPEEDUP: {speedup:.2f}x  |  Saved: {saved:.1f}ms/frame")
fps_old = 1000.0 / old_avg if old_avg > 0 else 0
fps_new = 1000.0 / wall_avg if wall_avg > 0 else 0
print(f"  Theoretical max FPS: {fps_old:.1f} -> {fps_new:.1f}")
print("=" * 72)
