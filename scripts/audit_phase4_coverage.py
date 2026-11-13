"""Measure how much identity evidence survives detection and local tracking."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from scripts.validate_phase4 import opportunities
from selfwatch_v2.contracts import FrameRef, Instance
from selfwatch_v2.evaluate_tracking import evaluate
from selfwatch_v2.tracking import LocalTracker
from selfwatch_v2.wildtrack_tracking import pair_scores


def audit(data, labels, source, *, pair_by_index=False):
    cases = ("all_detections", "camera_local", "motion_iou", "bytetrack")
    augmented = dict(data, tracking_cases={case: case for case in cases}, frames=[])
    motion, byte = {}, {}
    for frame in data["frames"]:
        ref = FrameRef(**frame["frame"])
        scope = ref.camera_id, ref.stream_epoch
        instances = tuple(Instance(**obs) for obs in frame["instances"])
        tracks = [dict(global_id=f"{frame['observation_id']}:{index}", local_id=index,
                       instance_index=index) for index, obs in enumerate(frame["instances"])
                  if obs["label"] == "person"]
        if scope not in motion:
            motion[scope] = LocalTracker()
            byte[scope] = BYTETracker(SimpleNamespace(track_high_thresh=.5, track_low_thresh=.1,
                new_track_thresh=.5, track_buffer=4, match_thresh=.8, fuse_score=True))
        motion_tracks = motion[scope].update(ref, instances)
        indices = [i for i, obs in enumerate(instances) if obs.label == "person"]
        boxes = np.asarray([[*instances[i].box_xyxy, instances[i].score, 0] for i in indices],
                           dtype=np.float32).reshape(-1, 6)
        byte_tracks = byte[scope].update(Boxes(boxes, orig_shape=(ref.height, ref.width)))
        byte_tracks = [dict(local_id=int(track[4]), instance_index=indices[int(track[-1])])
                       for track in byte_tracks]
        augmented["frames"].append(dict(frame, tracks_by_case=dict(
            frame["tracks_by_case"], all_detections=tracks, **{
                name: [dict(track, global_id=json.dumps([*scope, track["local_id"]]))
                       for track in assigned]
                for name, assigned in (("motion_iou", motion_tracks), ("bytetrack", byte_tracks))
            })))

    pair_labels = labels["frames"]
    if pair_by_index:
        pair_labels = [dict(row, frame=dict(row["frame"], captured_at_s=row["frame"]["frame_index"] / 10))
                       for row in pair_labels]
    times = {row["observation_id"]: row["frame"]["captured_at_s"] for row in pair_labels}

    results = {}
    for case in cases:
        result = evaluate(augmented, labels, case=case, reference_source=source)
        traces = [dict(row, captured_at_s=times[row["observation_id"]]) for row in result["frame_matches"]]
        slices = opportunities({"frames": pair_labels}, traces)
        results[case] = dict(
            matched_person_boxes=sum(len(row["matches"]) for row in traces),
            **({} if case == "all_detections" else dict(
                IDF1=result["IDF1"], sampled_global_id_changes=result["sampled_global_id_changes"])),
            **pair_scores(pair_labels, traces),
            episode_observability={
                name: {key: row[key] for key in ("opportunities", "anchor_available", "destination_matched")}
                for name, row in slices.items()
            },
            same_view_returns_recovered=slices["same_view_returns"]["recovered"],
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wildtrack", type=Path, required=True)
    parser.add_argument("--icsens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def run(root, source, paired):
        data, labels = (json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
                        for name in ("predictions", "annotations"))
        return audit(data, labels, source, pair_by_index=paired)

    output = dict(wildtrack=run(args.wildtrack, "official:wildtrack", False),
                  icsens=run(args.icsens, "official:icsens", True))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
