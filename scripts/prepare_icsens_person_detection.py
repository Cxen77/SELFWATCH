"""Build a sequence-separated YOLO person dataset from official i.c.sens labels."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile

import cv2

from scripts.validate_icsens_identity import GT_SHA256


PARTITIONS = {
    "train": (Path("datasets/icsens-identity-v1"), 0, tuple(range(0, 231, 5))),
    "val": (Path("datasets/icsens-phase4b-holdout-v1"), 6, tuple(range(0, 111, 5))),
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def marks(archive, camera, sequence):
    path = f"ground_truth/view{camera}/{sequence:04}.txt"
    result = {}
    for line in archive.read(path).decode().splitlines():
        if not line.strip():
            continue
        index, _subject, x, y, width, height, pedestrian, _occlusion, _ = line.split()
        if pedestrian == "1":
            result.setdefault(int(index), []).append(tuple(map(float, (x, y, width, height))))
    return result


def main():
    output = Path("datasets/icsens-person-detection-v1")
    if output.exists():
        raise FileExistsError(f"Preserve existing experiment data: {output}")
    output.mkdir(parents=True)
    first_gt = PARTITIONS["train"][0] / "ground_truth.zip"
    if digest(first_gt) != GT_SHA256:
        raise ValueError("Training labels differ from the pinned official archive")
    records, counts = [], Counter()
    with zipfile.ZipFile(first_gt) as archive:
        for split, (source, sequence, indices) in PARTITIONS.items():
            if digest(source / "ground_truth.zip") != GT_SHA256:
                raise ValueError(f"{split} labels differ from the pinned official archive")
            for camera in (1, 2):
                by_frame = marks(archive, camera, sequence)
                for index in indices:
                    relative = Path(f"view{camera}/{index:06}.png")
                    image_path = source / relative
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError(f"Unreadable image: {image_path}")
                    height, width = image.shape[:2]
                    destination = output / "images" / split / f"view{camera}_{index:06}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(image_path, destination)
                    label_path = output / "labels" / split / f"view{camera}_{index:06}.txt"
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    rows = []
                    for x, y, box_w, box_h in by_frame.get(index, ()):
                        x1, y1 = max(0., x), max(0., y)
                        x2, y2 = min(float(width), x + box_w), min(float(height), y + box_h)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        rows.append(f"0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                                    f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                    counts[split] += len(rows)
                    records.append(dict(split=split, sequence=sequence, camera=camera,
                        frame=index, image=str(destination), image_sha256=digest(destination),
                        label=str(label_path), label_sha256=digest(label_path),
                        width=width, height=height, boxes=len(rows)))
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: person\n", encoding="utf-8")
    manifest = dict(dataset="i.c.sens person detection adaptation, sequence-separated",
        source="https://doi.org/10.25835/0082741", ground_truth_sha256=GT_SHA256,
        partitions={split: {"sequence": spec[1], "indices": list(spec[2])}
                    for split, spec in PARTITIONS.items()},
        image_count={split: sum(row["split"] == split for row in records)
                     for split in PARTITIONS}, box_count=dict(counts), items=records,
        warning="Sequence 0007 remains a sealed evaluation set. These person boxes do not train hands, products, depth or ReID.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "images": manifest["image_count"],
                      "pedestrian_boxes": dict(counts), "manifest_sha256": digest(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
