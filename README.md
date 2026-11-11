# SELFWATCH

SELFWATCH tracks people across synchronized or timestamped camera views. It
combines person detection, local tracking, appearance matching and identity memory
to keep a shared Person ID across views. Body and hand overlays are optional.

Use the repaired [testing portal](docs/PORTAL_CONTROLS.md) at
http://127.0.0.1:8766/: separate video/camera/demo sources, one-click laptop
capture, inline Play/Pause/seek/step controls, analysis Pause/Resume/Stop, actual
processing FPS and person memory. The old 8765 address redirects to it.
Start `.\scripts\start-studio.ps1`. Progress and paper-writing evidence live in
the [master project log](docs/PROJECT_LOG.md); the current research stage is 2.

A frozen COCO-WholeBody diagnostic and CPU/CUDA hand-refinement comparison now
measure body/hand localization separately from detector misses. Predicted-ROI
refinement improved diagnostic hand PCK from 35.8% to 46.4%, but its confidence
scale and added latency require calibration/scheduling before live use. See
E15–E17 of the [master log](docs/PROJECT_LOG.md); the candidate remains offline.

The overhead person-detector pilot has now been trained and compared with its
pretrained baseline. It improved one 640-pixel diagnostic but regressed at 960
and failed ordinary-retail confidence/identity admission checks. It remains an
experimental branch; the portal retains its existing general model. See E14 in
the [master log](docs/PROJECT_LOG.md) for measured results, failure figures and
the model card. The [earlier overhead study](docs/OVERHEAD_DETECTOR_STUDY.md)
remains a historical report. Perception and identity accuracy gates remain open.

The [identity study](docs/IDENTITY_STUDY.md) compares three tracking policies on
14 paired retail clips. E19 in the master log adds a labelled three-camera
WILDTRACK pilot: it finds low cross-camera matching coverage even though the
frontend can display shared IDs. Retail tracking accuracy still needs its own
labels.

The [original-engine integration](docs/OLD_PROJECT_INTEGRATION.md) now combines
YOLO and whole-body perception with the original StrongSORT, phantom, warm-memory,
fingerprint, occlusion and ownership stack. This is the default review path;
shared Person IDs and memory states are visible. Start it with
`.\scripts\open-review.ps1`. The document lists active, advisory and retained
components, along with current verification and limitations.

V2 implementation has started on `codex/v2-replay-perception`: a recorded-video
YOLO runner with explicit detection/segmentation/pose outputs, camera-scoped
timestamps, provenance and durable observations. See [v2 usage](docs/V2_IMPLEMENTATION.md).
The historical performance claims below describe the old project and have not
been reproduced by the v2 implementation.

The [paired-camera review demo](docs/PAIRED_CAMERA_DEMO.md) adds RTMW whole-body
landmarks, local person tracks, reference model comparison, preliminary visual
labels and annotation export. It is a perception research prototype; the identity
update below adds appearance association, while item-custody reasoning remains later work.

The [ReID and CPU/GPU update](docs/REID_AND_SCHEDULING.md) now connects the old
OSNet encoder, global registry, cognitive galleries and attention scheduling.
It adds experimental cross-view appearance links and bounded CPU/GPU overlap;
item custody and calibrated cross-camera identity evaluation remain later stages.

SELFWATCH is a cognitive trajectory-first multi-object tracking system focused on persistent identity continuity, memory-augmented tracking, and occlusion-aware identity reasoning.

## Core Goals
- Reduce ID switching
- Maintain persistent identity during occlusion
- Improve temporal continuity
- Explore cognitive memory-inspired tracking
- Build trajectory-first identity ownership

## Current Stack
- RT-DETR Nano
- StrongSORT+
- ByteTrack-inspired recovery
- Cognitive Identity Layer
- Active/Warm Memory
- Phantom Tracking
- Occlusion Reasoning
- ACT-R inspired persistence

## Current Performance
- ~18.6 FPS on RTX 4060 Laptop GPU
- ~21ms detection latency
- Real-time capable pipeline

## Current Research Focus
- Trajectory-first ownership
- Occlusion-aware identity persistence
- Human-like identity continuity
- Cognitive memory systems
