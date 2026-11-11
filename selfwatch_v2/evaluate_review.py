"""Score person boxes only on explicitly exhaustive reviewed frames."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .tracking import iou


def evaluate(data, review):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if review["run_id"] != data["run_id"]:
        raise ValueError("Review refers to a different run")
    all_frames = {frame["observation_id"]: frame for frame in data["frames"]}
    variants = {"YOLO + RTMW": all_frames, **data.get("variants", {})}
    result = {name: Counter() for name in variants}
    issues = Counter()
    for annotation in review["frames"]:
        key = annotation["observation_id"]
        if key not in all_frames:
            raise ValueError("Unknown reviewed frame")
        issues.update(annotation.get("issues", []))
        if not annotation.get("reviewed") or "person" not in annotation.get("exhaustive_labels", []):
            continue
        boxes = [mark["xyxy"] for mark in annotation["marks"] if mark["label"] == "person" and mark["kind"] == "box"]
        for name, observations in variants.items():
            predictions = [p["box_xyxy"] for p in observations[key]["instances"] if p["label"].lower() == "person"]
            matches = 0
            if boxes and predictions:
                overlaps = np.array([[iou(a,b) for b in predictions] for a in boxes])
                # Maximize valid match cardinality first, overlap second.
                costs = np.where(overlaps >= .5, -1-overlaps*.001, 0)
                rows, cols = linear_sum_assignment(costs)
                matches = int(sum(overlaps[r,c] >= .5 for r,c in zip(rows,cols)))
            result[name].update(frames=1, ground_truth=len(boxes), predictions=len(predictions),
                                true_positive=matches, false_positive=len(predictions)-matches,
                                false_negative=len(boxes)-matches)
    reports = {}
    for name, counts in result.items():
        reports[name] = dict(counts)
        reports[name]["precision"] = counts['true_positive']/counts['predictions'] if counts['predictions'] else None
        reports[name]["recall"] = counts['true_positive']/counts['ground_truth'] if counts['ground_truth'] else None
    return dict(models=reports, issue_notes=dict(issues), iou_threshold=.5,
                scope="Selected reviewed frames only; person boxes. No hand/pose, tracking or retail accuracy claim.",
                annotation_source=review.get("annotation_source", "unspecified"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.data.read_text(encoding="utf-8")), json.loads(args.review.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
