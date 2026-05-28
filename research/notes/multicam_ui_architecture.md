# Multi-Camera UI Architecture

## Problem
The backend (`multicam/` module) was successfully upgraded to handle multi-stream cognitive tracking, but the frontend (`app.py`) remained tightly coupled to a single video feed, a single pipeline instance, and a single display panel. We needed a UI architecture that seamlessly scaled to N cameras without hacking the original frame-rendering logic.

## Hypothesis
We hypothesized that we could preserve the existing evaluator and layer-toggling features by:
1. Reusing the existing `VideoPlayer` component but instantiating a dynamic grid of them within `app.py`.
2. Extracting the color assignment logic in `engine/pipeline.py` to optionally accept a `color_map` (mapping local pipeline GIDs to global multi-cam GIDs). This ensures identical cross-camera color assignments directly within the pre-existing rendering pipeline.
3. Decoupling the `StateCache` from a single-frame constraint by allowing it to cache lists of frames and metadata per timestamp.

## Implementation
1. **Dynamic Grid Generation:** Removed the single `self.video_player` in `app.py` and replaced it with a dynamic `CTkFrame` container. Upon initializing a multi-camera session, the UI automatically creates a responsive grid of `VideoPlayer` instances based on the number of selected sources.
2. **Cross-Camera Coloring:** Modified `engine/pipeline.py`'s `process_frame` and internal rendering to check `color_map`. If provided, bounding box color seeds use the unified global ID rather than the local ID, achieving perfect visual continuity.
3. **Multi-Camera Timeline Scrubbing:** Upgraded `engine/state_cache.py` to encode and retrieve lists of frames. Scrubbing the timeline now synchronously scrubs all camera panels simultaneously.
4. **Live Event Logging:** Embedded a dedicated `CTkTextbox` into the `SidePanel` that listens directly to the `CameraEventBus` for real-time `ENTER`, `EXIT`, `MATCH`, and `NEW_GLOBAL` events.
5. **Evaluator Independence:** Kept the `SELFWATCHEvaluator` and `ResearchExperimentTracker` logic intact by executing evaluation independently for each active camera pipeline and appending suffixes to the output logs.

## Result
The SELFWATCH App now correctly visualizes multiple simultaneous streams in independent GUI panels. A person walking from Camera 0 to Camera 1 preserves their unique Global ID and retains their unique bounding box color. Forensic layers, memory layers, and motion predictions overlay correctly on a per-camera basis.

## Conclusion
The scalable UI architecture provides the foundational interface for evaluating Phase 2 (Topology & Transition modeling). The UI serves as a pure visualizer for the `MultiCameraPipeline` backend, keeping rendering logic clean and decoupled from multi-camera orchestration.
