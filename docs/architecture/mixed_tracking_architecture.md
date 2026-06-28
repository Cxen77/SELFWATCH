# Mixed Person-Object Tracking Architecture

## 1. Overview
The current SELFWATCH pipeline is highly optimized for person tracking. To support the Ownership Engine for retail theft detection, the pipeline must be upgraded to a **Mixed-Class Architecture**. This design allows the simultaneous tracking of humans and inanimate objects (backpacks, suitcases, laptops, etc.) while routing them through different association logic to prevent ReID corruption and maximize performance.

## 2. Tracking Strategy for Non-Person Objects

**Decision:** **ByteTrack-style (IoU/Geometry-only) Tracking**

**Why not ReID (OSNet)?**
OSNet is trained specifically on pedestrian datasets (Market-1501, DukeMTMC). Feeding inanimate objects into OSNet will produce arbitrary, unstable embeddings. This will destroy the Hungarian matching cost matrix and cause extreme identity switching for objects.

**Why not Motion-only?**
Objects held by people (phones, handbags) exhibit highly erratic, non-linear motion that breaks standard Kalman filter predictions. 

**The Solution:**
Objects will bypass the ReID feature extractor entirely. In the `StrongSORT` fused association stage, the `appearance_weight` for object classes will be hard-forced to `0.0`, reducing their matching purely to C-BIoU (Cascade-Buffered IoU) and ByteTrack's high/low confidence spatial matching.

## 3. Preserving `class_id` Across the Pipeline

To maintain class identity from pixels to the renderer, the data flow must be opened:

1.  **Detector (`rtdetr_detector.py`)**: Already natively outputs `class_ids`.
2.  **Pipeline Entry (`pipeline.py`)**: 
    - Expand `target_classes` to include `[0, 24, 26, 28, 63, 67]`.
    - Extract `det_result.class_ids` instead of discarding them.
3.  **ReID Bypass (`pipeline.py`)**:
    - Split bounding boxes into `person_boxes` and `object_boxes`.
    - Run `self.reid.extract()` *only* on `person_boxes`.
    - Inject zero-vectors for `object_boxes` to maintain array shapes for the tracker.
4.  **Tracker Interface (`strongsort_tracker.py`)**:
    - Add `class_ids` to `tracker.update(boxes, scores, class_ids, embeddings...)`.
    - Pass `class_id` to `STrack(..., class_id=cid)`.
5.  **Memory & Rendering (`strack.py` & `overlay_renderer.py`)**:
    - Store `self.class_id`.
    - Expose `class_id` in `_id_states` for the renderer to draw appropriate labels and icons.

## 4. Migration Plan (Safest Implementation Path)

We will execute the migration in four isolated, testable steps:

### Phase A: State Expansion (No logic changes)
1.  Update `STrack.__init__` to accept and store `class_id` (defaulting to 0).
2.  Update `tracker.update()` signature to accept `class_ids`.
3.  Update `engine/pipeline.py` to extract `class_ids` from the detector and pass them to the tracker.
*Validation: System runs identically, but tracks now possess semantic memory.*

### Phase B: ReID Bypass (Performance protection)
1.  In `engine/pipeline.py`, intercept the bounding boxes before they enter the ReID extractor.
2.  Filter out any box where `class_id != 0`.
3.  Pass only `PERSON_CLASS` boxes to OSNet.
4.  Reconstruct the `embeddings` array, placing the extracted features at the person indices and `np.zeros(512)` at the object indices.
*Validation: ReID extraction time drops, GPU memory is saved, no crashes occur.*

### Phase C: Tracker Association Update (Accuracy protection)
1.  In `strongsort_tracker.py`, inside `_fused_associate()`, identify tracks where `class_id != 0`.
2.  Force their appearance distance to `0.0` or force their appearance weight (`eff_app_w`) to `0.0`.
3.  Ensure the Kalman filter handles erratic object motion gracefully (potentially increasing process noise for object classes).
*Validation: Objects track stably using purely spatial overlap.*

### Phase D: Renderer & Ownership Engine Integration
1.  Expose `class_id` in `self._id_states`.
2.  Update `OverlayRenderer` to display class names (e.g., "Backpack" instead of "Person").
3.  Connect the `OwnershipManager` to process `PersonNode` and `ObjectNode` accurately.

## 5. FPS Impact Estimation

**Expected Impact: Neutral to Slight Improvement (+1 FPS)**

*   **Detector:** RT-DETR predicts all COCO classes natively in a single forward pass. Asking it to keep more classes merely changes a boolean mask during post-processing. **Cost: +0.0 ms.**
*   **ReID (OSNet):** By explicitly filtering out objects before ReID extraction, we *reduce* the batch size sent to the GPU compared to a naive implementation. **Cost: -0.0 ms** (since objects were previously ignored anyway).
*   **Tracker:** The Hungarian algorithm cost matrix will grow slightly (e.g., from 15x15 to 30x30). However, `scipy.optimize.linear_sum_assignment` solves 50x50 matrices in `<0.5 ms`. **Cost: +0.5 ms.**
*   **Overall:** The architectural decision to completely bypass ReID for non-person classes guarantees that tracking objects will incur zero GPU overhead, keeping the system strictly bound by the existing person-tracking baseline.
