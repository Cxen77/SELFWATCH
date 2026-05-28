# Multi-Camera Phase 1 — Implementation Notes

**Date:** 2026-05-27
**Category:** Notes
**Phase:** Phase 1

---

## Files Created

### New Package: `multicam/`

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Package exports | ~30 |
| `global_registry.py` | `GlobalMultiCameraIdentityManager` — shared ID space | ~350 |
| `cross_camera_reid.py` | `CrossCameraReIDMatcher` — baseline cosine matching | ~200 |
| `camera_stream.py` | `CameraStream` — per-camera processing wrapper | ~250 |
| `multicam_pipeline.py` | `MultiCameraPipeline` — top-level orchestrator | ~380 |
| `events.py` | `CameraEventBus` + `CameraEvent` — entry/exit tracking | ~150 |

### New Entry Point

| File | Purpose |
|------|---------|
| `multicam_main.py` | CLI entry point for multi-camera mode |

### Modified Files

| File | Change |
|------|--------|
| `config.py` | Added `MULTICAM_*` configuration section |

---

## Data Flow

```
Frame from VideoCapture
    │
    ▼
CameraStream.process_frame()
    │
    ├─ Runs local SelfWatchPipeline.process_frame()
    │      (detection, ReID, tracking, cognitive memory — unchanged)
    │
    ├─ Extracts active pipeline GIDs + embeddings
    │
    ├─ For NEW tracks:
    │      ├─ CrossCameraReIDMatcher.attempt_match()
    │      │      ├─ Compare embedding vs ALL dormant identities
    │      │      ├─ Temporal gating (max 300s gap)
    │      │      ├─ Confidence gating (min 0.05)
    │      │      └─ Return matched global_id or None
    │      │
    │      ├─ If match: reuse global_id, reactivate dormant
    │      └─ If no match: allocate new global_id
    │
    ├─ For EXISTING tracks:
    │      └─ Update observation in GlobalMultiCameraIdentityManager
    │
    ├─ For EXITED tracks (was active, now gone):
    │      ├─ Publish EXIT event
    │      ├─ Unregister from global registry
    │      └─ Move to dormant if not active elsewhere
    │
    └─ Annotate frame with CAM_ID | LID | GID
```

---

## Key Design Decisions

### 1. Round-Robin vs Threaded Processing
Chose round-robin for Phase 1. Simpler to debug, deterministic behavior.
Thread safety infrastructure is already in place for Phase 2.

### 2. Shared vs Independent Models
Shared detector + ReID, independent trackers.
Saves ~1.5GB VRAM per additional camera on RTX 4060.

### 3. Two-Level Identity Mapping
Each pipeline keeps its own internal identity numbering.
The multicam layer adds a second level of mapping.
This avoids any modification to the existing pipeline code.

### 4. Dormant Identity Window
300 seconds with exponential decay (rate=0.998).
~10 minute effective window before expiration.
Conservative but safe for Phase 1.

### 5. Event Bus Pattern
All cross-camera events flow through a central bus.
Decouples event producers (cameras) from consumers (ReID, logging, future topology).
Thread-safe with listener callbacks.

---

## Testing Strategy

1. **Unit test**: Create `GlobalMultiCameraIdentityManager`, register/unregister tracks, verify ID mapping
2. **Integration test**: Two video files, verify cross-camera matching
3. **Live test**: Two webcams, walk between camera views
4. **Stress test**: Run with 3+ sources, verify no crashes or race conditions

---

## Known Limitations

- FPS drops linearly with number of cameras (sequential processing)
- No camera topology learning
- No transition time prediction
- Embedding-only matching vulnerable to similar-looking people
- No cross-camera ambiguity resolution
- Grid display may be small with 3+ cameras
