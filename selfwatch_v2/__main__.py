"""Recorded-video perception runner; no camera capture or external alerts."""
import argparse
from collections import deque
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

from .replay import load_manifest, replay
from .store import ObservationStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--keypoint-schema", help="Explicit topology, e.g. coco17")
    parser.add_argument("--max-frames", type=int, default=None, help="Total across cameras")
    parser.add_argument("--preview", action="store_true", help="Save annotated JPEGs")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    sources = load_manifest(args.manifest)
    from .yolo import YoloPerception
    model = YoloPerception(args.weights, args.device, args.imgsz, args.confidence, args.keypoint_schema)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "sources": [vars(source) for source in sources],
              "model": model.provenance,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("ultralytics", "torch", "numpy", "opencv-python")}}
    (args.output / "run.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    timings, count, per_camera = deque(maxlen=10000), 0, {}
    camera_directories = {source.camera_id: f"camera-{index}" for index, source in enumerate(sources)}
    stream = replay(sources)
    started = time.perf_counter()
    status = "failed"
    try:
        with ObservationStore(args.output / "observations.sqlite") as store:
            for frame, image in stream:
                observation, result = model.predict(frame, image)
                store.append(observation)
                timings.append(observation.inference_ms)
                count += 1
                per_camera[frame.camera_id] = per_camera.get(frame.camera_id, 0) + 1
                if args.preview:
                    import cv2
                    directory = args.output / "preview" / camera_directories[frame.camera_id]
                    directory.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(directory / f"{frame.frame_index:08d}.jpg"), result.plot()):
                        raise RuntimeError("Failed to write preview")
                if args.max_frames is not None and count >= args.max_frames:
                    break
        status = "complete"
    finally:
        stream.close()
        elapsed = time.perf_counter() - started
        ordered = sorted(timings)
        summary = {"status": status, "frames": count, "per_camera": per_camera,
            "wall_seconds": elapsed, "aggregate_processed_fps": count / elapsed if elapsed else 0,
            "inference_p50_ms": statistics.median(timings) if timings else None,
            "inference_p95_ms": ordered[max(0, int(len(ordered) * .95 + .999) - 1)] if ordered else None,
            "timing_samples": len(timings),
            "timing_note": "Last 10000 inference samples; cold start included when retained. Offline throughput, not live capture latency or accuracy."}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
