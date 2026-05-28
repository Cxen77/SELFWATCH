# Multi-Camera Phase 1 Architecture — Design Diary

**Date:** 2026-05-27
**Author:** SELFWATCH Research
**Phase:** Phase 1 — Foundational Multi-Camera Infrastructure

---

## Objective

Run 2+ video streams simultaneously, maintaining ONE shared global identity space across all cameras. Build clean foundational infrastructure WITHOUT over-engineering.

## Architecture Decision: Layered Separation

### Why NOT a monolithic multi-camera tracker?

The existing SELFWATCH pipeline is a complex cognitive tracking system with:
- RT-DETR detection
- StrongSORT tracking with fused costs
- Global identity management with inertia
- Ambiguity freezing / occlusion groups
- Territorial ownership / phantom tracking
- Cognitive reasoning and persistence

Merging multi-camera logic into this pipeline would create unmaintainable coupling. Instead, we chose a **layered wrapper architecture**:

```
Camera Pipelines (independent, existing code)
    ↓
Shared Global Memory (new: GlobalMultiCameraIdentityManager)
    ↓
Cross-Camera Identity Assignment (new: CrossCameraReIDMatcher)
    ↓
Future Cognitive Layer (Phase 2+)
```

### Key Design Principle

> Each camera runs its own full SelfWatchPipeline.
> The multi-camera layer wraps around them, not inside them.

This preserves:
1. All existing single-camera cognitive logic (untouched)
2. Per-camera local tracker independence (parallel-safe)
3. Clean separation of concerns (local vs global identity)
4. Future scalability (add cameras without modifying pipeline)

---

## Identity Namespace Strategy

### Problem: Two Identity Spaces

Each `SelfWatchPipeline` has its own `GlobalIdentityManager` producing local "global" IDs. With N cameras, these IDs collide:
- Camera0 GID 5 ≠ Camera1 GID 5

### Solution: Two-Level ID Mapping

1. **Pipeline-local GID**: Internal to each camera's pipeline (unchanged)
2. **Multicam Global ID**: True cross-camera global identity

Mapping: `(camera_id, pipeline_gid) → multicam_global_id`

The `CameraStream` maintains this mapping and synchronizes on every frame.

---

## Dormant Identity State

### Why dormant instead of delete?

When a person exits Camera A, they may appear in Camera B seconds or minutes later. Deleting the identity would force creating a new global ID, fragmenting identity.

### Design:
- **Active**: Currently visible in at least one camera
- **Dormant**: Not visible anywhere, but retained with embeddings
- **Expired**: Confidence decayed below threshold, archived

Dormant identities:
- Retain full embedding gallery (up to 10 embeddings)
- Retain EWMA-smoothed stable embedding
- Decay exponentially (rate=0.998 per tick)
- Are searchable by the cross-camera ReID matcher
- Have a minimum confidence threshold (0.05) before expiration
- Track last-seen camera and time for temporal gating

---

## Cross-Camera ReID: Baseline Approach

### Algorithm (Phase 1 — Intentionally Simple)

```
1. Person disappears from Camera A → DORMANT
2. New track appears in Camera B
3. Compare embedding against ALL dormant identities
4. If cosine_similarity(query, dormant.stable_embedding) ≥ 0.70:
     → Reuse same global ID
5. Apply temporal gating:
     - Reject if time_gap > 300 seconds
6. Apply confidence gating:
     - Reject if dormant.confidence < 0.05
```

### Why cosine similarity only?

Phase 1 is a **baseline**. More sophisticated matching (topology graphs, gait signatures, color histograms) will be added in Phase 2+. The baseline establishes the infrastructure and data flow.

### Gallery Matching

We compare against both:
- The EWMA-smoothed stable embedding (most robust)
- All gallery embeddings (captures appearance variation)

Take the maximum similarity as the score.

---

## Thread Safety Model

All shared state is protected by `threading.Lock`:
- `GlobalMultiCameraIdentityManager`: All public methods are locked
- `CameraEventBus`: Thread-safe publish/consume
- `CrossCameraReIDMatcher`: Match log is locked

Phase 1 uses **synchronous round-robin** processing (simpler, debuggable).
Phase 2+ can add per-camera threads with the same locking infrastructure.

---

## GPU Resource Sharing

### Problem: N cameras × (detector + ReID) = N× GPU memory

### Solution: Shared models, independent trackers

```
Shared (1 instance):
    - RT-DETR detector
    - OSNet ReID extractor

Independent (N instances):
    - StrongSORTTracker
    - SelfWatchPipeline (cognitive memory, state machines)
```

This means detection and ReID run sequentially across cameras, but each camera's tracking state is fully independent. On a single GPU, this is optimal since the GPU can only run one model at a time anyway.

---

## Limitations of Phase 1

1. **Sequential processing**: Cameras are processed round-robin, reducing effective FPS per camera
2. **No topology**: No spatial transition model (which cameras connect?)
3. **No temporal modeling**: No prediction of transition times
4. **No cross-camera ambiguity resolution**: No handling of similar-looking people across cameras
5. **Naive embedding matching**: Gallery comparison only, no learned cross-camera metric
6. **No camera calibration**: No homography or world-coordinate mapping
7. **Shared model bottleneck**: Detection runs sequentially across cameras

---

## Recommended Next Phase (Phase 2)

1. **Threaded per-camera processing**: Each camera in its own thread
2. **Camera topology graph**: Learn which cameras are connected
3. **Transition time modeling**: Expected transit times between cameras
4. **Appearance adaptation**: Domain-adaptive embedding normalization per camera
5. **Cross-camera cognitive layer**: Extend ambiguity freezing across cameras
6. **Zone-based reasoning**: Entry/exit zones per camera for transition prediction
