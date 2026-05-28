# Phase 1 + 2 Optimization — Research Diary

## Date: 2026-05-28

---

## PHASE 1: Async Forensic Logger / State Cache Decoupling

### Problem
`state_cache.append()` was called inside the inference thread every frame.
It ran `cv2.imencode('.jpg', frame, quality=70)` for every camera's frame before handing control back to the GPU. This cost 4–10ms of synchronous CPU work per frame — invisible in the profiling HUD because it sat between the inference timer end and the atomic handoff.

### Implementation
Created `engine/async_state_cache.py`:
- Inference thread calls `append()` → posts raw numpy frames to a bounded `queue.Queue(maxsize=4)` → returns **immediately**.
- Background daemon thread (`sw-state-encoder`) compresses frames from the queue.
- Scrubbing, playback, and `get_frame()` behavior preserved exactly.
- Uses PyTurboJPEG if available, falls back to OpenCV.
- New diagnostics line: `Enc: Xms | EncQ: Y | EncDrops: Z`

### Rollback
Swap `from engine.async_state_cache import AsyncStateCache` back to `from engine.state_cache import StateCache` in `app.py` line 15 and line 35.

### Expected Gain
- Removes 4–10ms from inference hot path per frame per camera
- Inference cadence becomes more uniform (no burst JPEG latency)

---

## PHASE 2: Tracker Cost Matrix Vectorization

### Problem
`StrongSORTTracker._fused_associate()` computed the (N×M) cost matrix using nested Python for loops:
```python
for i in range(n_det):          # 15 iterations
    for j in range(n_trk):      # 50 iterations = 750 total
        math.hypot(...)         # Python scalar
        exit_trajectories.get() # Python dict lookup × 750
        ...
```
With 15 detections × 50 tracks = 750 Python loop iterations per call.
The function was called up to 3 times per frame (Stages 1A, 1B, 2).
Total Python overhead: ~25–35ms per camera frame.

### Implementation
Replaced the full inner loop body with NumPy broadcasting:

| Operation | Before | After |
|---|---|---|
| IoU matrix | `iou_matrix()` call (was ok) | `iou_matrix()` unchanged |
| Cosine distance | `det_embs @ track_emb_matrix.T` (ok) | Same, now assigned in one shot |
| Trajectory centers | Per-cell dict lookup × 750 | Pre-built arrays `traj_cx[M]`, `traj_cy[M]` in O(M) pass |
| Center distance | `math.hypot()` × 750 | `np.sqrt(diff_cx**2 + diff_cy**2)` broadcasting (N,M) |
| Effective appearance weight | Per-cell if/else | `np.where(is_frozen[None,:], 0.0, ...)` vectorized (N,M) |
| Frozen gate | Per-cell `if is_frozen` | `np.where(frozen_gate_mask, GATE_VALUE, cost)` |
| Cooldown corridor gate | Per-cell dict + `math.hypot` | Vectorized distance comparison |
| Cooldown vel gate | Per-cell dot product | Vectorized `dot_num / dot_denom` |
| Depth quantization | `for i × for j` | `np.abs(det_depth[:,None] - trk_depth[None,:])` |
| Spatial/appearance gate | Per-cell conditionals | `np.where` masked arrays |

**Cross-partner isolation** remains a Python loop — it's an inherently sparse operation (only frozen/cooldown tracks with partners) and scales O(partners × n_trk), not O(N×M).

### Cognitive Correctness
All constraints preserved:
- ✅ Frozen track: `eff_app_w = 0.0` + spatial gate at 45px
- ✅ Cooldown trajectory commitment cost (0.6 × app + 0.4 × dist)
- ✅ Cooldown corridor gating (50px / 120px if strong visual)
- ✅ Cooldown velocity direction gate (dot < 0.6 / -0.2)
- ✅ Cooldown appearance gate (app_dist > 0.28)
- ✅ Cross-partner crossing rejection
- ✅ Pseudo-depth quantization (+0.5 / +0.15 penalty)
- ✅ Zero-IoU fast-motion rescue (appearance + velocity + distance)
- ✅ Bad appearance gate
- ✅ Crowd disambiguation gate
- ✅ AMI (Hungarian post-filter)

### Rollback
The previous `_fused_associate` body was ~400 lines starting at line 478. Revert git commit to restore.

### Expected Gain
- Tracker latency: 27ms → 3–6ms (4–9× speedup)
- FPS improvement: potentially +3–5 FPS end-to-end per camera

### Test Results (Post-Phase 1 & 2)
The user executed the multi-camera tracking test after these changes.
**Measured Metrics:**
- **Overall FPS:** 8.0 - 9.2 FPS (7.3 FPS average over full session including startup)
- **Detector latency:** ~18.54 ms [GPU]
- **ReID latency:** ~6.97 ms [GPU] (Average over frames)
- **Tracker latency:** ~17.16 ms [CPU] (Down from 27ms)
- **Total Frame latency:** ~54.63 ms per camera (109.2ms combined = ~9.15 FPS)

**Analysis:**
The tracker latency successfully dropped from 27ms to 17ms, validating the vectorization optimizations. The background caching (Phase 1) successfully stabilized the execution loop without dropping forensic capabilities. We are now hovering around ~9 FPS for 2 cameras.
While the O(N×M) loops were vectorized, the tracker still consumes 17ms, likely due to remaining non-vectorized components like ByteTrack IoU stages and C-BIoU fallback stages which have not been vectorized yet. 
The next big performance jumps will come from:
1. **Phase 5 (Multi-Camera Batched Detection):** Running the detector once for all cameras as a batched tensor, drastically reducing GPU launch overhead.
2. **Phase 3 (Render Decoupling):** Further decoupling `cv2.resize` and `cv2.putText` away from the hot path.

---

## PHASE 3: Render Decoupling & UI Pipeline Stabilization

### Problem
Before Phase 3, the `engine/pipeline.py` performed extensive OpenCV drawing (bounding boxes, trajectories, labels, HUD, forensic diagnostics) directly inside the inference thread. Additionally, `app.py` performed expensive `cv2.cvtColor` and `cv2.resize` operations every 33ms tick, even when the frame hadn't changed. This caused:
- Inference stuttering (inference thread blocked by `cv2.polylines` and `cv2.putText`).
- UI threading bottlenecks (duplicate resize operations eating Tkinter thread time).
- Jittery camera output since the display could only advance when inference was completely done drawing.

### Implementation
1. **`OverlayRenderer` Decoupling:**
   - Removed all OpenCV drawing logic from `engine/pipeline.py`.
   - `pipeline.py` now populates a lightweight `rendering_metadata` dictionary.
   - Created `ui/overlay_renderer.py` which takes the raw frame + metadata and applies all annotations (boxes, HUD, forensic layers).
   - The drawing is now executed purely inside `app.py`'s `_display_tick()` or `_update_ui_from_cache()` methods.
2. **Stale Frame Skipping:**
   - Updated `VideoPlayer.update_frame()` to cache `self._last_frame_id`.
   - If the same numpy array is submitted multiple times, the UI skips the expensive `cv2.cvtColor`, `cv2.resize`, and PIL conversion completely.
3. **Resize & Display Optimizations:**
   - `VideoPlayer` caches target dimensions on `<Configure>` events, avoiding slow Tkinter `winfo_width()` IPC queries on the hot path.
   - BGR→RGB conversion is now a fast numpy slice (`[::-1]`).
4. **Independent Metrics:**
   - `app.py` now tracks `_render_fps` completely separate from `_inference_fps`.

### Rollback
Revert git commit `Phase 3: render decoupling - VideoPlayer caching + render FPS diagnostics` and `Phase 3: Decoupled OverlayRenderer implementation` to restore synchronous drawing.

### Expected Gain
- Removes ~2–5ms of OpenCV drawing overhead from the inference hot path.
- Tkinter UI remains buttery smooth at 30FPS regardless of inference drops.
- `VideoPlayer` CPU usage drops drastically when inference runs < 30FPS (due to skipped duplicate frames).
