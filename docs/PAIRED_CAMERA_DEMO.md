# Paired-camera whole-body review

Implemented 1 November 2026 on `codex/v2-replay-perception`.

The completed sample is served locally at http://127.0.0.1:8765/ . Its portable output folder is `runs/paired-review-000398-v2/`. The earlier `runs/paired-review-000398/` folder is an interrupted CPU experiment, not the completed demo.

## What works

- Recorded-time playback of 32 frames per camera, including original uneven sampling intervals.
- Model switch between YOLO segmentation + RTMW, YOLO17 pose, SH17 body-part detection and YOLO segmentation alone on identical source images.
- RTMW 133-point COCO whole-body output: body, feet, face and both 21-landmark hands. Scores remain model scores, not calibrated visibility probabilities.
- Local motion/IoU Hungarian tracking with capture-time velocity, camera/epoch-scoped IDs, expiry and observed-only trajectory overlays. This is a transparent baseline, not a crossing/occlusion guarantee or cross-camera identity solution.
- Overlay controls, paired timeline, frame age, frame stepping, playback speed and observation list.
- Manual person/head/hand/item boxes, wrist points, subject references, observation notes, undo, review state and JSON export. Browser-local changes require export to persist as a separate file.
- Person-box evaluation at IoU 0.5 only when the annotator explicitly marks person annotations exhaustive. Unreviewed or partial frames cannot silently become negative ground truth.

Verification: 23 automated tests pass in the dedicated environment. The browser interface was checked in desktop and narrow layouts, including model switching, timestamp stepping, playback, canvas point annotation, undo, reversible reviewed state and JSON download. The downloaded browser export was parsed and scored successfully, preserving all 64 frame records, the six preliminary labels and their assistant attribution. Browser console error inspection returned no errors during the check.

## Runtime and model

The project now has a dedicated `.venv` using Python 3.11, CUDA torch 2.11.0+cu128 and Ultralytics 8.4.194. Runtime versions are locked in `requirements-v2-lock.txt`; bootstrap is `scripts/install-v2.ps1`. RTMLib 0.0.16 is installed with its declared dependencies; both OpenCV distributions are held to the same version. `pip check` passed.

The RTMW lightweight export was downloaded from OpenMMLab's official host using the URL listed by RTMLib's whole-body configuration. Checkpoint hash `68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76` and archive provenance are recorded under `weights/v2/rtmw/`. This is the `rtmw-dw-l-m` export with a 192×256 input. Do not relabel it as an unverified different architecture.

YOLO runs on the RTX 4060; whole-body pose explicitly uses ONNX Runtime CPU for this first validated run. Requested unavailable CUDA pose raises an error rather than silently falling back. An initial default ORT CPU session oversubscribed threads and was interrupted; a bounded two-thread CPU session completed the paired run in approximately 23.3 seconds, excluding model construction. This is not a live FPS benchmark or a fully GPU-optimized deployment.

The run produced 56 person observations and 56 corresponding whole-body pose arrays across 64 frames. This does not mean 56 unique shoppers or 56 correct detections. Pose is conditioned on detector boxes and cannot recover people the detector missed.

## First visual review and error report

Six raw frames were visually inspected separately from overlays: camera ranks 0 and 1 at frame indices 0, 16 and 30. Approximate person boxes were recorded as preliminary assistant annotations. They require independent human adjudication before research use. Hidden/ambiguous hand coordinates were deliberately left unassigned.

`initial-review.json` contains these labels and source attribution. They load into the review interface with an explicit preliminary status. `error-report.json` compares the four outputs against these six boxes. Its numbers are a debugging aid on a tiny selected sample, not a publishable model ranking. The larger 64-frame clip is still mostly unreviewed.

Concrete findings:

- One camera misses a clearly visible shopper in some frames even when the other view detects them.
- At frame 16, the primary detector yields no person in rank0 and two person detections in rank1, although the inspected images contain one visible shopper per view. Extra pose landmarks do not resolve detector misses or duplicate hypotheses.
- The local baseline emits one ID in rank0 and two IDs over time in rank1. Without track-level ground truth, this is an indicator to investigate, not a measured identity accuracy result.
- The SH17 checkpoint's explicit head/ear/hands taxonomy does not make its person boxes reliably transfer to top-down retail views. Gloves predictions must not be equated with bare-hand detections.
- No quantitative finger/wrist accuracy has been established. Further pose labels and visibility checks are required before choosing or fine-tuning a hand model.

These findings support the next comparison: annotate more views and occlusions; inspect person miss/duplicate regimes; compare resolution and domain tuning; measure whether selective high-resolution hand inference improves visible-hand localization. Shared identity memory is still a separate next stage.

## Run and review

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m selfwatch_v2.demo --manifest datasets/retailaction-replay-verified/000398.json --output runs/new-paired-review --comparison "YOLO17=runs/retailaction-000398-pose" --comparison "SH17 parts=runs/retailaction-000398-ppe" --comparison "YOLO segmentation=runs/retailaction-000398-seg"
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory runs/paired-review-000398-v2
```

Use a fresh output directory when rebuilding. `scripts/open-review.ps1` also launches the completed review. If port 8765 is already occupied, use a different port; do not stop unrelated processes. The server is bound to loopback.

Export review JSON from the screen, then evaluate:

```powershell
.\.venv\Scripts\python.exe -m selfwatch_v2.evaluate_review --data runs/paired-review-000398-v2/data.json --review PATH_TO_EXPORTED_REVIEW.json --output runs/review-evaluation.json
.\.venv\Scripts\python.exe -m unittest discover -s tests_v2 -v
```

Raw data, labels, previews and weights are ignored by git. The completed review includes RetailAction's LICENSE and NOTICE alongside its derivative images. Preserve dataset/model terms before sharing artifacts. No camera feed, POS data, real-world identity or theft label is inferred here.
