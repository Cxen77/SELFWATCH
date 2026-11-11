# First v2 verification

Date: 1 November 2026. Branch: `codex/v2-replay-perception`.

Implemented a recorded-video runner, event-time ordering, explicit per-checkpoint task outputs, immutable SQLite observations, checkpoint/package provenance and optional annotated JPEGs. Legacy application behavior is not used by this runner.

Ten foundation tests passed under Python 3.11. Coverage includes camera/epoch identity collisions, timestamp ordering, bounded merge buffering, missing cameras, nonmonotonic input rejection, duplicate-source rejection, observation retries/conflicts, custom class names, pose topology mismatch and segmentation pixel coordinates.

Real GPU inference passed for YOLO26n pose (8 frames), YOLO26n segmentation (8 frames), and the existing fuzzy SH17 YOLOv8 detector (4 frames). Input was a deliberately synthetic two-source recording made from a repeated Ultralytics sample image with a 50 ms offset. Both cameras contributed observations. Saved records were inspected for nonempty pose/segmentation outputs. This is an integration test, not synchronized-store validation, a retail accuracy test or a speed benchmark. Startup and JPEG writing substantially affect these tiny runs.

The inference runtime used for these checks was the existing read-only fuzzy virtual environment: `C:/Users/Dev/Desktop/fuzzy/jetson-safety/.venv/Scripts/python.exe`. No packages, data or models in that project were changed. Observed packages: torch 2.11.0+cu128, ultralytics 8.4.194, OpenCV 5.0.0; CUDA was available. A dedicated v2 environment is still to be provisioned and locked for reproducible deployment.

Explicitly downloaded checkpoints in `weights/v2/`: YOLO26n pose and YOLO26n segmentation, obtained from the official Ultralytics assets v8.4.0 release. Per-run hashes are in the ignored `runs/` artifacts. Those assets carry their own licensing terms; the repository's license does not override them.

Fuzzy checkpoint inspection confirmed that `sh17_yolov8n.pt` has explicit `person`, `ear`, `hands`, `head`, `face` and other PPE-related classes. Its SHA-256 is `5449f62bd7cf30c01c7fd138db2729a38abcb4d5120449773fba475d54f30b62`. That explains its ability to produce body-part boxes absent from a generic COCO taxonomy; it does not establish accuracy on retail footage.

Artifacts (ignored by git): `runs/v2-pose-smoke/`, `runs/v2-seg-smoke/`, `runs/v2-ppe-smoke/`, `runs/v2-smoke-inputs/`, `runs/fuzzy-checkpoints.json`. These contain manifests, observations, summaries and annotated previews. Re-run with a new output directory, using the command format in `V2_IMPLEMENTATION.md`.

Remaining work: dedicated runtime setup, real recorded clips/annotations, whole-body/hand inference, calibration, global identity/custody memory, and POS reconciliation. No shoplifting or payment accuracy is claimed by this milestone.
