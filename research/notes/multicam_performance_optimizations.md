# Real-Time Multi-Camera Pipeline Optimization

## Problem
In our multi-camera phase 1 architecture, the introduction of a second camera caused FPS to plummet to 3-4. Videos became heavily delayed, RAM spiked to 90%, while GPU usage remained surprisingly low (~40%). The pipeline exhibited classic signs of blocking I/O, unbounded buffer growth, and sequential frame backlogs.

## Hypothesis
We hypothesized that the bottleneck wasn't hardware limits (RTX 4060 can easily run RT-DETR + ReID for 2 cameras). Instead, the bottlenecks were:
1. **OpenCV Read Blocking & Buffering:** Synchronous `read()` calls fall behind real-time, causing OpenCV to buffer old frames. When finally processed, the pipeline works on ancient frames.
2. **Unbounded Memory Accumulation:** `StateCache`, `ActiveMemory` lists, and Evaluator histories were accumulating infinitely. High-res images in RAM caused massive bloat.
3. **Detector Over-polling:** Running RT-DETR natively at 30 FPS across multiple cameras is wasteful when StrongSORT can confidently predict bounding boxes between frames.

## Implementation
1. **Async Threaded Frame Reader:**
   Replaced synchronous `cap.read()` in `CameraStream` with an asynchronous `_reader_loop` thread. It uses an internal queue (`_frame_queue`) with a hard cap of 2. If the pipeline is slower than the camera, the reader instantly drops the oldest frame, guaranteeing the pipeline only ever processes the *freshest* frame.
2. **Early Resizing:**
   Injected a `cv2.resize(frame, (960, 540))` at the very top of `pipeline.process_frame()`. This exponentially reduces the memory footprint of all downstream `StateCache` frames and rendering operations without degrading human-perceivable debugging quality. ReID internal cropping to 128x128 remains unaffected.
3. **Detector Skipping (Temporal Interleaving):**
   Introduced `DETECTOR_INTERVAL=2` in `engine/pipeline.py`. RT-DETR now only runs every other frame. On skipped frames, the pipeline bypasses detection and feeds StrongSORT's previous `smooth_box` predictions back into itself. This maintains identity continuity seamlessly while instantly halving GPU detector workload.
4. **Memory Capping:**
   - Reduced `StateCache` from an enormous 2000 frames to 500.
   - Enforced hard limits on evaluator `switch_events` history (max 500).
   - Enforced limits on `ActiveMemory` trajectory and confidence histories (max 30).
5. **UI Diagnostics & Evaluator Toggle:**
   Added an `[EVALUATION ON/OFF]` checkbox in `app.py` to bypass metric calculations (which scale with `O(N^2)` on identities) in production. Added real-time diagnostics displaying RAM usage, per-camera Queue Sizes, and dropped frame counts.

## Result
Pipeline execution throughput effectively doubled. The async reader completely eliminated the video delay issue, creating a true real-time display. Dropped frames are now gracefully handled by the reader thread rather than bogging down the inference loop. RAM usage stabilized at significantly lower margins due to early 960x540 resizing and bounded queues.

## Conclusion
Real-time multi-camera tracking requires decoupling I/O ingestion from GPU inference, prioritizing latency over absolute frame-by-frame completeness. The implementation of "latest-frame-only" queuing and interval-based detector skipping proved essential for restoring cognitive tracking to real-time.
