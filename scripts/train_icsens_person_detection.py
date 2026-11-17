"""Train a research-only person detector on seq 0000; validate on seq 0006."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import traceback


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO

    dataset = Path("datasets/icsens-person-detection-v1")
    weights = Path("weights/v2/yolo26n.pt")
    output = Path("runs/icsens-person-training-v1")
    if output.exists():
        raise FileExistsError(f"Preserve prior experiment: {output}")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["partitions"]["train"]["sequence"] != 0 or manifest["partitions"]["val"]["sequence"] != 6:
        raise ValueError("Training/validation sequence lock differs")
    for row in manifest["items"]:
        if digest(Path(row["image"])) != row["image_sha256"]:
            raise ValueError("Training image differs from manifest")
        if digest(Path(row["label"])) != row["label_sha256"]:
            raise ValueError("Training label differs from manifest")

    recipe = dict(data=str((dataset / "dataset.yaml").resolve()), epochs=20,
        imgsz=640, batch=4, device=0, workers=0, cache=False, amp=True,
        seed=42, deterministic=True, patience=5, degrees=180.0,
        fliplr=0.5, flipud=0.5, mosaic=0.5, close_mosaic=5,
        plots=False, save=True, project=str(output.parent.resolve()),
        name=output.name, exist_ok=False)
    output.mkdir(parents=True)
    lock = dict(status="starting", started_utc=datetime.now(timezone.utc).isoformat(),
        protocol="Train only on i.c.sens sequence 0000; validation/checkpoint selection on sequence 0006; keep sequence 0007 sealed.",
        dataset_manifest_sha256=digest(manifest_path),
        initial_weights={"path": str(weights.resolve()), "sha256": digest(weights)},
        recipe=recipe, hardware={"gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "ultralytics": ultralytics.__version__})
    lock_path = output / "experiment.json"
    def persist():
        lock_path.write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")
    persist()
    try:
        torch.set_num_threads(2)
        model = YOLO(str(weights.resolve()))
        if model.task != "detect":
            raise ValueError("Expected a detection checkpoint")
        lock["status"] = "training"
        persist()
        recipe["exist_ok"] = True
        model.train(**recipe)
        best = output / "weights" / "best.pt"
        lock["status"] = "completed"
        lock["best_checkpoint"] = {"path": str(best), "sha256": digest(best), "bytes": best.stat().st_size}
    except Exception:
        lock["status"] = "failed"
        lock["error"] = traceback.format_exc()
        raise
    finally:
        lock["ended_utc"] = datetime.now(timezone.utc).isoformat()
        persist()


if __name__ == "__main__":
    main()
