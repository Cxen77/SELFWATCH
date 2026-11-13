"""Audit whether a higher-resolution detector can supply both handoff endpoints."""
import argparse
from collections import defaultdict
from pathlib import Path
import tempfile

import scripts.validate_icsens_identity as icsens
from scripts.validate_phase4 import frozen_write
from scripts.validate_phase4_continuity import read
from selfwatch_v2.evaluate_tracking import evaluate
from selfwatch_v2.identity_study import digest
from selfwatch_v2.wildtrack_tracking import pair_scores
from selfwatch_v2.yolo import YoloPerception


def endpoint_audit(data, labels):
    frames = []
    for row in data["frames"]:
        tracks = [dict(global_id=f"{row['observation_id']}:{i}", local_id=i, instance_index=i)
                  for i, item in enumerate(row["instances"]) if item["label"] == "person"]
        frames.append(dict(row, tracks_by_case={"detections": tracks}))
    predictions = dict(data, frames=frames, tracking_cases={"detections": "Detector ceiling"})
    result = evaluate(predictions, labels, case="detections", reference_source=icsens.SOURCE)
    by_id = {row["observation_id"]: row["frame"] for row in labels["frames"]}
    visible, detected = defaultdict(lambda: defaultdict(set)), defaultdict(set)
    for row in labels["frames"]:
        frame = row["frame"]
        for mark in row["marks"]:
            visible[mark["subject"]][frame["frame_index"]].add(frame["camera_id"])
    traces = []
    for row in result["frame_matches"]:
        frame = by_id[row["observation_id"]]
        index, camera = frame["frame_index"], frame["camera_id"]
        traces.append(dict(row, captured_at_s=index / 10))
        for match in row["matches"]:
            detected[match["subject"]].add((index, camera))
    episodes = []
    for subject, history in visible.items():
        previous, previous_views = None, set()
        for index, views in sorted(history.items()):
            if previous is not None and index - previous <= 20 and not previous_views & views:
                for destination in views:
                    anchor = any((previous, camera) in detected[subject] for camera in previous_views)
                    target = False
                    for step in range(index, index + 21, 5):
                        if destination not in history.get(step, ()):
                            break
                        target |= (step, destination) in detected[subject]
                    episodes.append((anchor, target))
            previous, previous_views = index, views
    paired = [dict(row, frame=dict(row["frame"], captured_at_s=row["frame"]["frame_index"] / 10))
              for row in labels["frames"]]
    return dict(
        matched_boxes=sum(len(row["matches"]) for row in traces),
        handoff_opportunities=len(episodes),
        source_detected=sum(source for source, _ in episodes),
        destination_detected=sum(target for _, target in episodes),
        both_detected=sum(source and target for source, target in episodes),
        **pair_scores(paired, traces),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline, labels = read(args.baseline)
    scores = {"640": endpoint_audit(baseline, labels)}
    detector_factory = icsens.YoloPerception

    def at_960(weights, device, _size, confidence):
        return YoloPerception(weights, device, 960, confidence)

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "run"
        try:
            icsens.YoloPerception = at_960
            icsens.audit(args.dataset, path, sequence=6)
        finally:
            icsens.YoloPerception = detector_factory
        data, checked_labels = read(path)
        if checked_labels != labels:
            raise ValueError("The detector comparison must use the same reference annotations")
        scores["960"] = endpoint_audit(data, labels)
        frozen_write(args.output, dict(scope="Post-holdout detector diagnostic; i.c.sens 0006, IoU .50.",
            scores=scores, labels_sha256=digest(args.baseline / "annotations.json"),
            baseline_predictions_sha256=digest(args.baseline / "predictions.json"),
            high_resolution_predictions_sha256=digest(path / "predictions.json")))


if __name__ == "__main__":
    main()
