"""Build a local paired-view review package from actual perception outputs."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import time
from hashlib import sha256

from .replay import load_manifest, replay
from .store import ObservationStore
from .tracking import LocalTracker
from .wholebody import WholeBody
from .yolo import YoloPerception
from .appearance import AppearanceEncoder
from .identity import SharedIdentity
from .pipeline import process
from .legacy_guard import LegacyIdentityGuard


def write_review_assets(output):
    """Version assets so reusing the review port cannot show the previous UI."""
    assets = Path(__file__).parent / "review_assets"
    html=(assets/'index.html').read_text(encoding='utf-8')
    for name in ('review.js','review.css'):
        payload=(assets/name).read_bytes()
        (output/name).write_bytes(payload)
        html=html.replace(f'"{name}"',f'"{name}?v={sha256(payload).hexdigest()[:12]}"')
    (output/'index.html').write_text(html,encoding='utf-8')


def build(args):
    import cv2
    import torch
    torch.set_num_threads(2)
    if args.pose_device=='cuda':
        from .gpu_runtime import activate
        activate()
    sources = load_manifest(args.manifest)
    if len(sources) != 2:
        raise ValueError("This review screen requires a paired-camera manifest")
    if args.max_frames < 1:
        raise ValueError("Positive frame limit required")
    detector = YoloPerception(args.weights, args.device, args.imgsz, args.confidence)
    rotation_mode=getattr(args,'person_rotation_mode','off')
    if rotation_mode!='off':
        from .rotation import RotationPersonPerception
        detector=RotationPersonPerception(detector,mode=rotation_mode)
    pose = WholeBody(args.pose_weights, args.pose_device, args.max_people)
    torch_device="cpu" if args.device=="cpu" else args.device if args.device.startswith("cuda:") else f"cuda:{args.device}"
    appearance = AppearanceEncoder(args.reid_weights,device=torch_device)
    identity = SharedIdentity(similarity=args.reid_threshold,margin=args.reid_margin,
                              allow_overlap=args.allow_overlap)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    tracker_engine=getattr(args,'tracker_engine','legacy')
    trackers = {s.camera_id: LegacyIdentityGuard(log_dir=args.output/'cognitive_logs')
                if tracker_engine=='legacy' else LocalTracker() for s in sources}
    frames, counts, totals = [], Counter(), Counter()
    timing_records=[]
    stream = replay(sources)
    started = time.perf_counter()
    run_config = dict(manifest=json.loads(args.manifest.read_text(encoding="utf-8")),
        detector=detector.provenance, pose=pose.provenance, reid=appearance.provenance, settings=vars(args),
        packages={name: importlib.metadata.version(name) for name in
                  ("torch", "ultralytics", "onnxruntime", "rtmlib", "opencv-python", "scipy")})
    (args.output / "run.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    processed=None
    try:
        with ObservationStore(args.output / "observations.sqlite") as store:
            import itertools
            processed=process(itertools.islice(stream,args.max_frames),detector,pose,trackers,
                              appearance,identity,parallel=not args.serial and args.pose_device=='cpu')
            for observation,image,tracks,diagnostics,timings in processed:
                frame=observation.frame
                store.append(observation)
                filename = f"images/{len(frames):06d}.jpg"
                write_started=time.perf_counter()
                if not cv2.imwrite(str(args.output / filename), image):
                    raise RuntimeError("Cannot write review image")
                frames.append(dict(observation.to_dict(), image=filename, tracks=tracks,
                                   observation_id=frame.observation_id,appearance_diagnostics=diagnostics,
                                   identity_memory=timings.pop('identity_memory',[]),
                                   identity_audit=timings.pop('identity_audit',{}),
                                   perception_diagnostics=timings.pop('perception_diagnostics',None)))
                timings["image_write_ms"]=(time.perf_counter()-write_started)*1000
                timing_records.append(dict(observation_id=frame.observation_id,**timings))
                counts[frame.camera_id] += 1
                totals.update(person_observations=len(tracks),
                    posed_person_observations=sum(i.keypoints_xyc is not None for i in observation.instances))
    finally:
        if processed is not None:
            processed.close()
        stream.close()
        for tracker in trackers.values():
            if hasattr(tracker,'close'):
                tracker.close()
    data = dict(schema_version=1, name="SELFWATCH paired-view lab", sample=args.manifest.stem,
        run_id=detector.provenance["sha256"][:12]+"-"+pose.provenance["sha256"][:12]+"-"+args.output.name,
        cameras=[s.camera_id for s in sources], frames=frames, pose=pose.provenance,
        timeline_semantics="Recorded sample timestamps; latest available frame per view, not invented synchronization",
        identity=identity.snapshot(),
        tracker_engine=tracker_engine,
        identity_semantics="Experimental appearance links with abstention, not ground-truth identity; local IDs remain camera-scoped")
    variants = {}
    for specification in args.comparison:
        label, directory = specification.split("=", 1)
        observations = {}
        with sqlite3.connect(Path(directory) / "observations.sqlite") as database:
            for (payload,) in database.execute("SELECT payload FROM observations"):
                record = json.loads(payload)
                f = record["frame"]
                key = json.dumps([f["camera_id"], f["stream_epoch"], f["frame_index"]], separators=(",", ":"))
                observations[key] = record
        matched = {}
        for frame in frames:
            other = observations.get(frame["observation_id"])
            if other:
                if other["frame"] != frame["frame"]:
                    # Receipt time is inherently different between repeated runs.
                    comparable = ("source", "captured_at_s", "width", "height")
                    if any(other["frame"][field] != frame["frame"][field] for field in comparable):
                        raise ValueError("Comparison observations refer to different source frames")
                matched[frame["observation_id"]] = other
        if len(matched) != len(frames):
            raise ValueError(f"Comparison {label} lacks frames; refusing misleading comparison")
        variants[label] = matched
    data["variants"] = variants
    (args.output / "data.json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    (args.output / "timings.json").write_text(json.dumps(timing_records,indent=2),encoding="utf-8")
    (args.output / "identity.json").write_text(json.dumps(identity.snapshot(),indent=2),encoding="utf-8")
    write_review_assets(args.output)
    summary = dict(frames=len(frames), per_camera=dict(counts), **totals,
                   wall_seconds=time.perf_counter()-started, annotation_status="Not manually reviewed",
                   accuracy="Not measured; ground-truth annotations required", providers=pose.provenance["providers"],
                   cpu_gpu_overlap=not args.serial and args.pose_device=='cpu', reid_links=len(identity.events),
                   cross_camera_links=sum(e.get("association_kind")=="cross_camera" for e in identity.events),
                   host_process_rss_mb=__import__('psutil').Process().memory_info().rss/1024**2,
                   peak_torch_reserved_mb=torch.cuda.max_memory_reserved(torch_device)/1024**2 if args.device!="cpu" else 0,
                   memory_note="Torch process allocations only; excludes other GPU processes and drivers")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights/v2/yolo26n-seg.pt"))
    parser.add_argument("--pose-weights", type=Path, default=Path("weights/v2/rtmw/wholebody133.onnx"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pose-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--max-people", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--comparison", action="append", default=[], help="Label=existing-run-directory")
    parser.add_argument("--reid-weights",type=Path,default=Path("weights/osnet/osnet_x1_0_msmt17.pth"))
    parser.add_argument("--reid-threshold",type=float,default=.78)
    parser.add_argument("--reid-margin",type=float,default=.06)
    parser.add_argument("--allow-overlap",action="store_true",help="Permit active identities in other views as candidates")
    parser.add_argument("--serial",action="store_true",help="Disable CPU/GPU overlap for controlled comparison")
    parser.add_argument("--tracker-engine",choices=('legacy','baseline'),default='legacy',
                        help="Original full cognitive stack (default) or earlier motion baseline for ablation")
    parser.add_argument('--person-rotation-mode',choices=('off','weak','memory','always'),default='off',
                        help='Optional same-model person recovery; extra GPU passes, default off')
    build(parser.parse_args())


if __name__ == "__main__":
    main()
