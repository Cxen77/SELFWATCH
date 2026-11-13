"""Evaluate existing shared identity on an independent i.c.sens camera pair."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import torch

from scripts.validate_phase4 import opportunities
from selfwatch_v2.appearance import AppearanceEncoder
from selfwatch_v2.contracts import FrameRef
from selfwatch_v2.evaluate_tracking import evaluate
from selfwatch_v2.identity_study import Trial, digest
from selfwatch_v2.wildtrack_tracking import pair_scores
from selfwatch_v2.yolo import YoloPerception


SOURCE = "official:icsens"
INDEXES = tuple(range(0, 231, 5))
HANDOFF_INDEXES = tuple(range(0, 111, 5))
GT_SHA256 = "10e36036ceb02de4bf9dfcdaa285e136251777982197d6c376bc9b7dd9af9e00"


def read_marks(archive, camera, sequence=0):
    result = defaultdict(list)
    rows = archive.read(f"ground_truth/view{camera}/{sequence:04}.txt").decode().splitlines()
    for line in rows:
        if not line.strip():
            continue
        index, subject, x, y, width, height, pedestrian, occlusion, _ = line.split()
        if pedestrian != "1":
            continue
        x, y, width, height = map(float, (x, y, width, height))
        result[int(index)].append(dict(subject=subject, box=(x, y, x + width, y + height),
                                       occlusion=float(occlusion)))
    return result


def audit(root, output, *, sequence=0):
    indices = HANDOFF_INDEXES if sequence == 6 else INDEXES
    gt = root / "ground_truth.zip"
    if digest(gt) != GT_SHA256:
        raise ValueError("i.c.sens labels differ from the pinned archive")
    with zipfile.ZipFile(gt) as archive:
        labels = {c: read_marks(archive, c, sequence) for c in (1, 2)}
    clocks = {c: list(map(float, (root / f"view{c}/time_stamp.csv").read_text().splitlines()))
              for c in labels}
    if any(len(times) != indices[-1] + 1 or any(b <= a for a, b in zip(times, times[1:]))
           for times in clocks.values()):
        raise ValueError("Unexpected i.c.sens timing")
    skew = max(abs(clocks[1][i] - clocks[2][i]) for i in indices)
    if skew > .05:
        raise ValueError("Paired camera timestamps differ by over 50 ms")
    frames = sorted((clocks[c][i], c, i) for c in labels for i in indices)
    origin = frames[0][0]
    source = [SimpleNamespace(camera_id=f"view{c}") for c in labels]
    torch.set_num_threads(2)
    detector = YoloPerception("weights/v2/yolo26n-seg.pt", "0", 640, .25)
    detector.include_masks = False
    encoder = AppearanceEncoder("weights/osnet/osnet_x1_0_msmt17.pth")
    output.mkdir(parents=True, exist_ok=False)
    run_id = f"icsens:{sequence:04}:stride5"
    raw, annotations, image_hashes, vectors = [], [], {}, {}
    trial = Trial("legacy", source)
    try:
        with (output / "engine.log").open("w", encoding="utf-8") as log:
            for capture, c, index in frames:
                relative = f"view{c}/{index:06}.png"
                path = root / relative
                image_hashes[relative] = digest(path)
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Unreadable image {relative}")
                height, width = image.shape[:2]
                frame = FrameRef(f"view{c}", f"icsens-{sequence:04}", index, capture - origin, capture - origin,
                                 width, height, relative, "published timestamp")
                with redirect_stdout(log):
                    obs, _ = detector.predict(frame, image)
                    features, _, _ = encoder.extract_all(obs, image)
                    tracks = trial.step(obs, image, features)["tracks"]
                local = [dict(track, global_id=json.dumps([frame.camera_id, track["local_id"]]))
                         for track in tracks if track.get("global_id") is not None]
                keys = {str(i): f"f{len(raw)}_i{i}" for i in features}
                vectors.update({keys[str(i)]: v.copy() for i, v in features.items()})
                row = dict(obs.to_dict(), observation_id=frame.observation_id, feature_keys=keys,
                           tracks_by_case=dict(camera_local=local, existing_shared_memory=tracks))
                raw.append(row)
                marks = []
                for mark in labels[c][index]:
                    x1, y1, x2, y2 = mark["box"]
                    box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
                    if box[2] > box[0] and box[3] > box[1]:
                        marks.append(dict(label="person", kind="box", source=SOURCE,
                                          subject=mark["subject"], xyxy=box))
                annotations.append(dict(observation_id=frame.observation_id, frame=row["frame"],
                                        reviewed=True, exhaustive_labels=["person"], marks=marks))
                if len(raw) % 10 == 0:
                    print(f"{len(raw)}/{len(frames)} i.c.sens observations", flush=True)
    finally:
        trial.close()

    data = dict(run_id=run_id, frames=raw,
                tracking_cases={name: name for name in raw[0]["tracks_by_case"]})
    reference = dict(run_id=run_id, frames=annotations)
    scores = {}
    # Use source timestamps for causal inference. Group metrics by the paired
    # frame index only after verifying the maximum timestamp disagreement.
    paired_annotations = [
        dict(row, frame=dict(row["frame"], captured_at_s=row["frame"]["frame_index"] / 10))
        for row in annotations
    ]
    paired_times = {row["observation_id"]: row["frame"]["captured_at_s"] for row in paired_annotations}
    for case in data["tracking_cases"]:
        result = evaluate(data, reference, case=case, reference_source=SOURCE)
        traces = [dict(row, captured_at_s=paired_times[row["observation_id"]])
                  for row in result["frame_matches"]]
        scores[case] = dict(
            IDF1=result["IDF1"], labelled_person_boxes=sum(len(row["marks"]) for row in annotations),
            sampled_global_id_changes=result["sampled_global_id_changes"],
            **pair_scores(paired_annotations, traces),
            **opportunities({"frames": paired_annotations}, traces),
        )
    report = dict(source="https://doi.org/10.25835/0082741", ground_truth_sha256=GT_SHA256,
                  scope=f"Independent outdoor camera pair, sequence {sequence:04}, left images at every fifth frame.",
                  label_policy="Source pedestrian boxes; cyclists excluded from reference. Extra predictions charged.",
                  verified_view_skew_max_s=skew, observations=len(raw), scores=scores,
                  detector=detector.provenance, reid=encoder.provenance,
                  image_sha256=image_hashes,
                  timestamps_sha256={c: digest(root / f"view{c}/time_stamp.csv") for c in labels})
    for name, content in (("results", report), ("predictions", data), ("annotations", reference)):
        (output / f"{name}.json").write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "features.npz", **vectors)
    print(json.dumps(scores, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", type=int, choices=(0, 6), default=0)
    args = parser.parse_args()
    audit(args.dataset, args.output, sequence=args.sequence)


if __name__ == "__main__":
    main()
