"""Build audited per-sample replay manifests without modifying source labels."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def prepare(root, output):
    import cv2
    root, output = Path(root).resolve(), Path(output).resolve()
    provenance = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    if provenance["status"] != "complete":
        raise ValueError("Partial downloads cannot be prepared as complete pilots")
    for record in provenance["files"]:
        content = (root / record["path"]).read_bytes()
        if len(content) != record["size"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Corrupt downloaded file: {record['path']}")
    output.mkdir(parents=True, exist_ok=False)
    samples, excluded = [], []
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        content = metadata["content"]
        cameras = content["action_cam"]
        if any(not cameras[name].get("frame_timestamps") for name in ("rank0", "rank1")):
            excluded.append(dict(sample=metadata_path.parent.name, reason="Missing original frame timestamps; timing cannot be invented"))
            continue
        # Anonymized dates belong to each clip only, not a global store clock.
        all_dates = [datetime.fromisoformat(t) for camera in cameras.values() for t in camera["frame_timestamps"]]
        origin = min(all_dates)
        sources, frame_counts = [], {}
        for name in ("rank0", "rank1"):
            video = metadata_path.parent / f"{name}_video.mp4"
            times = [(datetime.fromisoformat(t) - origin).total_seconds() for t in cameras[name]["frame_timestamps"]]
            capture = cv2.VideoCapture(str(video))
            count = 0
            fps = capture.get(cv2.CAP_PROP_FPS)
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot decode {video}")
                while capture.read()[0]:
                    count += 1
            finally:
                capture.release()
            if count != len(times):
                raise ValueError(f"Frame/annotation mismatch: {video}: {count} vs {len(times)}")
            source = dict(camera_id=name, stream_epoch=f"retailaction-train-{metadata_path.parent.name}",
                          path=str(video), fps=fps, timestamps_s=times)
            # Validate the same contract the inference runner will consume.
            from .replay import Source
            Source(name, video, source["stream_epoch"], fps, timestamps_s=tuple(times))
            sources.append(source)
            frame_counts[name] = count
        manifest_path = output / f"{metadata_path.parent.name}.json"
        manifest = dict(schema_version=1, sources=sources, annotation_path=str(metadata_path),
                        split=provenance["split"], dataset_revision=provenance["revision"],
                        note="Nonuniform preselected frames; no continuous-video or full-visit claim")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        samples.append(dict(sample=metadata_path.parent.name, replay_manifest=str(manifest_path),
                            annotation_path=str(metadata_path), frames=frame_counts))
    index = dict(dataset=provenance["repository"], revision=provenance["revision"],
                 split=provenance["split"], selection=provenance["selection"], samples=samples,
                 excluded=excluded, exclusion_note="Timing-filtered pilot; not an unbiased benchmark")
    (output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = prepare(args.root, args.output)
    print(f"Verified {len(index['samples'])} paired samples with explicit frame timestamps; excluded {len(index['excluded'])}")


if __name__ == "__main__":
    main()
