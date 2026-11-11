# V2 recorded-video perception

Run from the SELFWATCH repository root. The new `selfwatch_v2` package avoids importing the legacy tracker and RF-DETR dependencies. It records camera-scoped observations for the forthcoming shared memory comparison. It does not yet assign visit IDs or determine item custody.

Create a JSON manifest with explicit constant-frame-rate recordings, offsets in seconds and a stable recording epoch. Paths resolve relative to the manifest. Offsets must be measured; default zero does not establish synchronization. For nonuniform sampled video, provide `timestamps_s` per source, one original event timestamp per decoded frame; these override frame-index/FPS timing. Count mismatch or nonmonotonic timestamps cause an error. RTSP timing is not yet supported. See [the RetailAction pilot](RETAILACTION_PILOT.md) for downloaded paired clips and the dataset importer.

```json
{
  "schema_version": 1,
  "sources": [
    {"camera_id": "aisle", "stream_epoch": "session-001", "path": "aisle.mp4", "fps": 30, "offset_s": 0},
    {"camera_id": "checkout", "stream_epoch": "session-001", "path": "checkout.mp4", "fps": 30, "offset_s": 0.12}
  ]
}
```

```powershell
python -m selfwatch_v2 --manifest recordings/session.json --weights weights/yolo26n-seg.pt --device 0 --output runs/v2-seg --max-frames 100 --preview
python -m selfwatch_v2 --manifest recordings/session.json --weights weights/yolo26n-pose.pt --keypoint-schema coco17 --device 0 --output runs/v2-pose --max-frames 100 --preview
python -m unittest discover -s tests_v2 -v
```

Supply an existing local checkpoint; automatic weight downloads are deliberately not implicit. Use a fresh output directory for each run. Segmentation and pose are separate runs/checkpoints here, not incorrectly merged person lists. No claim of finger detection is made for COCO17 pose. Custom classes retain their model-local IDs and names. Instance masks are serialized as contour polygons, not lossless full masks.

Outputs: `run.json` includes checkpoint hash, class taxonomy, package versions and configuration; `observations.sqlite` stores immutable observations; `summary.json` reports processed frames by camera and cold-start-inclusive runtime; optional annotated JPEGs are under `preview/`. Receipt timestamps represent replay decode on this host, not the original camera receipt time. Inference timing includes materializing output tensors on CPU. Offline FPS is not a live latency or accuracy benchmark.

The fuzzy project was located at `C:/Users/Dev/Desktop/fuzzy/jetson-safety`. Its inference script defaults to `models/sh17_yolov8n.pt`, a specialized PPE model. Its head labels therefore cannot be attributed to general COCO detection alone. Keep its checkpoints and data intact. Record the actual checkpoint names/hashes/classes before using it as a comparison.

The whole-body adapter, initial local tracker, paired-camera review and annotation export are now implemented. See [the paired demo](PAIRED_CAMERA_DEMO.md). Next stages include expanded annotated comparison, calibrated geometry, global uncertain identity links, revisable event/custody memory and POS reconciliation. Refer to the execution plan in the Desktop research directory for the evaluation gates.
