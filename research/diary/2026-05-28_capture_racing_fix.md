# Pipeline Stabilization — Critical Fix: File Capture Racing + Parallel Processing

## Date: 2026-05-28

## Problem
Multi-camera pipeline with 2 video files showed 3-9 FPS that froze at 4.1 FPS after a few seconds. Video stuck permanently.

## Root Cause Analysis

### Bug 1: Capture Thread Racing Through Video Files
The capture thread had `time.sleep(0.005)` for file sources, meaning it decoded frames at ~200fps. But inference ran at ~5fps. With `deque(maxlen=1)`, 97% of decoded frames were silently dropped. The capture thread exhausted the entire video file in seconds, set `_stream_ended = True`, and the inference thread stopped — **freezing the video**.

For a 2-minute video at 30fps (3600 frames):
- Capture decoded all 3600 frames in ~18 seconds
- Inference processed only ~90 frames in those 18 seconds  
- Result: video "ended" at frame 90 out of 3600

### Bug 2: Sequential Camera Processing
`step()` processed cameras in a for-loop. With profiling:
- CAM0: 180ms (detect 36ms + reid 91ms + tracker 35ms + 18ms other)  
- CAM1: 96ms (detect 34ms + reid 13ms + tracker 35ms + 14ms other)
- Sequential total: 180 + 96 = **276ms → 3.6 FPS**

The GPU sat idle during each camera's 35ms CPU tracker phase while the other camera waited its turn.

## Fix

### Fix 1: Back-Pressure Capture Pacing
For file sources, the capture thread now uses **back-pressure**: it waits until the inference thread has consumed the current frame before decoding the next one.

```python
if is_file:
    while len(self._frame_buf) >= 1 and self._capture_running:
        time.sleep(0.005)  # Wait for inference to consume
```

Result: video plays at inference speed, every frame is available, no premature exhaustion.

### Fix 2: Parallel Camera Processing via ThreadPoolExecutor
`step()` now uses `concurrent.futures.ThreadPoolExecutor` to process all cameras simultaneously. CPU work (tracker, reasoning, draw) overlaps across camera threads. GPU calls serialize naturally via CUDA's default stream.

Theoretical improvement:
- Before: CAM0(180ms) + CAM1(96ms) = 276ms → 3.6 FPS
- After: max(CAM0, CAM1) + GPU contention = ~200ms → **5+ FPS**

## Conclusion
The "freeze" was a file-reading race condition, not a memory leak or performance issue. The low FPS was caused by unnecessary sequential processing of independent cameras.
