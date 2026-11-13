"""Develop and independently score Phase 4 continuity candidates."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

from scripts.download_wildtrack_pilot import IDENTITY_CONTINUITY_HOLDOUT
from scripts.validate_phase4 import (
    GATE, frozen_write, geometry_inputs, opportunities, passing, replay, score,
)
from selfwatch_v2.appearance_motion import AppearanceMotion, LocalPolicy
from selfwatch_v2.contracts import FrameRef, Instance
from selfwatch_v2.evaluate_tracking import evaluate
from selfwatch_v2.identity_study import digest
from selfwatch_v2.overlap_memory import LinkPolicy, Sighting
from selfwatch_v2.transit_memory import TransitMemory, TransitPolicy
from selfwatch_v2.wildtrack_geometry import DEVELOPMENT, load_cameras, load_run
from selfwatch_v2.wildtrack_tracking import pair_scores


SOURCES = ("scripts/validate_phase4_continuity.py", "scripts/validate_phase4.py",
           "scripts/validate_icsens_identity.py", "selfwatch_v2/appearance_motion.py",
           "selfwatch_v2/transit_memory.py", "selfwatch_v2/overlap_memory.py")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in SOURCES}


def read(root):
    return [json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("predictions", "annotations")]


def tracked(data, vectors, settings, *, preserve_owners=False):
    trackers, frames = {}, []
    owners = {}
    for row in sorted(data["frames"], key=lambda r: (r["frame"]["captured_at_s"], r["frame"]["camera_id"])):
        frame = FrameRef(**row["frame"])
        if frame.camera_id not in trackers:
            trackers[frame.camera_id] = AppearanceMotion(settings)
        instances = tuple(Instance(**item) for item in row["instances"])
        features = {int(i): vectors[k] for i, k in row["feature_keys"].items()}
        results = trackers[frame.camera_id].update(frame, instances, features)
        if preserve_owners:
            scope = frame.camera_id, frame.stream_epoch
            original = [t for t in row["tracks_by_case"]["camera_local"]
                        if not t.get("visual_suppressed") and t.get("status", "observed") == "observed"]
            legacy_by_instance = {t["instance_index"]: t["local_id"] for t in original}
            reserved = set(legacy_by_instance.values())
            blocked = {t["instance_index"] for t in row["tracks_by_case"]["camera_local"]
                       if t.get("visual_suppressed")}
            restored = []
            for item in results:
                index = item["instance_index"]
                key = (*scope, item["local_id"])
                if index in legacy_by_instance:
                    owners[key] = legacy_by_instance[index], frame.captured_at_s
                    continue
                if index in blocked:
                    continue
                claim = owners.get(key)
                if claim is None or frame.captured_at_s - claim[1] > settings.max_gap_s:
                    continue
                owner = claim[0]
                if owner in reserved:
                    continue
                reserved.add(owner)
                restored.append(dict(item, local_id=owner))
            results = original + restored
            owners = {key: claim for key, claim in owners.items()
                      if frame.captured_at_s - claim[1] <= settings.max_gap_s}
        local = [dict(track, global_id=json.dumps([frame.camera_id, frame.stream_epoch, track["local_id"]]))
                 for track in results]
        frames.append(dict(row, tracks_by_case=dict(row["tracks_by_case"], camera_local=local)))
    return dict(data, frames=frames)


def icsens_scores(data, labels, case):
    result = evaluate(data, labels, case=case, reference_source="official:icsens")
    paired = [dict(row, frame=dict(row["frame"], captured_at_s=row["frame"]["frame_index"] / 10))
              for row in labels["frames"]]
    times = {row["observation_id"]: row["frame"]["captured_at_s"] for row in paired}
    traces = [dict(row, captured_at_s=times[row["observation_id"]]) for row in result["frame_matches"]]
    return dict(IDF1=result["IDF1"], changes=result["sampled_global_id_changes"],
                **pair_scores(paired, traces), **opportunities({"frames": paired}, traces))


def transit(data, vectors, settings):
    memory = TransitMemory([("view1", "view2"), ("view2", "view1")], settings)
    bundles = {}
    for row in data["frames"]:
        bundles.setdefault(row["frame"]["frame_index"], []).append(row)
    frames = []
    for _, rows in sorted(bundles.items()):
        latest = max(row["frame"]["captured_at_s"] for row in rows)
        sightings = []
        owners = {}
        for row in rows:
            frame = row["frame"]
            for track in row["tracks_by_case"]["camera_local"]:
                key = (frame["camera_id"], frame["stream_epoch"], track["local_id"])
                feature_key = row["feature_keys"].get(str(track["instance_index"]))
                sightings.append(Sighting(key, feature=vectors[feature_key] if feature_key else None))
                owners[row["observation_id"], track["local_id"]] = key
        ids = memory.update(latest, sightings)
        for row in rows:
            tracks = [dict(track, global_id=ids[owners[row["observation_id"], track["local_id"]]])
                      for track in row["tracks_by_case"]["camera_local"]]
            frames.append(dict(row, tracks_by_case=dict(row["tracks_by_case"], transit=tracks)))
    return dict(data, frames=frames, tracking_cases=dict(data["tracking_cases"], transit="Dormant owner handoff"))


def choose_local(data, labels, vectors):
    settings = [LocalPolicy(cosine=cosine, birth_score=birth, appearance_weight=weight, max_gap_s=gap)
                for cosine, birth, weight, gap in product((.48, .6, .72), (.27, .36, .45), (1.4, 2.6), (1., 1.75))]
    baseline = score(data, labels, "camera_local")
    rows = []
    for policy, preserve in product(settings, (False, True)):
        stats = score(tracked(data, vectors, policy, preserve_owners=preserve), labels, "camera_local")
        rows.append(dict(policy=asdict(policy), preserve_owners=preserve, score=stats))
    # Select on continuity first, break ties by paired observability.
    winner = max(rows, key=lambda row: (
        row["score"]["IDF1"] - .002 * max(0, row["score"]["sampled_global_id_changes"]
                                        - baseline["sampled_global_id_changes"]),
        row["score"]["same_person_both_matched"]))
    return winner, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "evaluate"))
    for name in ("wildtrack", "calibration", "icsens", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    development = args.mode == "fit"
    indices = DEVELOPMENT if development else IDENTITY_CONTINUITY_HOLDOUT
    data, labels, _ = load_run(args.wildtrack, indices)
    cameras, camera_sha = load_cameras(args.calibration)
    public, public_labels = read(args.icsens)
    expected = "icsens:0000:stride5" if development else "icsens:0006:stride5"
    if public["run_id"] != expected:
        raise ValueError("Unexpected handoff partition")
    baseline = score(data, labels, "camera_local")
    with np.load(args.wildtrack / "features.npz", allow_pickle=False) as cache, \
            np.load(args.icsens / "features.npz", allow_pickle=False) as public_cache:
        vectors = dict(cache)
        public_vectors = dict(public_cache)
        if development:
            winner, grid = choose_local(data, labels, vectors)
            selected = LocalPolicy(**winner["policy"])
            changed = tracked(data, vectors, selected, preserve_owners=winner["preserve_owners"])
            rig, bundles = geometry_inputs(changed, cameras, indices, vectors)
            overlaps = []
            for radius, margin, cosine, revise in product((1.1, 1.5), (.015, .05), (.45, .60), (False, True)):
                policy = LinkPolicy(radius_m=radius, margin_m=margin, min_cosine=cosine, appearance_weight_m=2.5)
                predictions, events, _ = replay(changed, rig, bundles, policy, revise=revise)
                stats = score(predictions, labels, "overlap", events)
                overlaps.append(dict(policy=asdict(policy), revise=revise, score=stats))
            eligible = [row for row in overlaps if (row["score"]["matched_pair_precision"] or 0) >= GATE["min_pair_precision"]]
            if not eligible:
                raise ValueError("No development overlap setting reached the precision floor")
            overlap = max(eligible, key=lambda row: (row["score"]["same_person_shared_id"], row["score"]["IDF1"]))
            cross_winner = max(
                (dict(policy=asdict(settings), preserve_owners=preserve,
                      score=icsens_scores(tracked(public, public_vectors, settings, preserve_owners=preserve),
                                          public_labels, "camera_local"))
                 for settings, preserve in product((selected, LocalPolicy(.60, .28, 2.6, 1.75),
                                                    LocalPolicy(.60, .36, 2.6, 1.75)), (False, True))),
                key=lambda row: (row["score"]["same_view_returns"]["recovered"], row["score"]["IDF1"]))
            cross = tracked(public, public_vectors, LocalPolicy(**cross_winner["policy"]),
                            preserve_owners=cross_winner["preserve_owners"])
            transits = []
            for cosine, margin, confirm in product((.64, .72, .80), (.05, .10), (1, 2)):
                policy = TransitPolicy(cosine=cosine, margin=margin, confirmations=confirm)
                stats = icsens_scores(transit(cross, public_vectors, policy), public_labels, "transit")
                transits.append(dict(policy=asdict(policy), score=stats))
            handoff = max(transits, key=lambda row: (
                row["score"]["nonoverlap_view_transitions"]["recovered"],
                -row["score"]["different_person_shared_id"], row["score"]["IDF1"]))
            frozen_write(args.policy, dict(local=winner, overlap=overlap, transit=handoff,
                transit_local=cross_winner, gate=GATE, source_sha256=source_hashes(),
                calibration_sha256=camera_sha, wildtrack_development_sha256=digest(args.wildtrack / "predictions.json"),
                icsens_development_sha256=digest(args.icsens / "predictions.json")))
            frozen_write(args.output, dict(baseline=baseline, local_grid=grid,
                                          overlap_grid=overlaps, transit_grid=transits))
            return

        locked = json.loads(args.policy.read_text(encoding="utf-8"))
        if (locked["calibration_sha256"] != camera_sha or locked["gate"] != GATE
                or locked["source_sha256"] != source_hashes()):
            raise ValueError("Calibration, source or gate differs from frozen policy")
        selected = LocalPolicy(**locked["local"]["policy"])
        changed = tracked(data, vectors, selected, preserve_owners=locked["local"]["preserve_owners"])
        rig, bundles = geometry_inputs(changed, cameras, indices, vectors)
        predicted, events, timing = replay(changed, rig, bundles,
            LinkPolicy(**locked["overlap"]["policy"]), revise=locked["overlap"]["revise"])
        overlap = score(predicted, labels, "overlap", events)
        cross = tracked(public, public_vectors, LocalPolicy(**locked["transit_local"]["policy"]),
                        preserve_owners=locked["transit_local"]["preserve_owners"])
        handoff = icsens_scores(transit(cross, public_vectors, TransitPolicy(**locked["transit"]["policy"])),
                               public_labels, "transit")
        gate = passing(dict(overlap, nonoverlap_view_transitions=handoff["nonoverlap_view_transitions"],
                            same_view_returns=handoff["same_view_returns"]), baseline)
        frozen_write(args.output, dict(gate=gate, passed=all(gate.values()), timing=timing,
            wildtrack_baseline=baseline, wildtrack_new_local=score(changed, labels, "camera_local"),
            wildtrack_overlap=overlap, icsens_baseline=icsens_scores(public, public_labels, "existing_shared_memory"),
            icsens_new_local=icsens_scores(cross, public_labels, "camera_local"), icsens_transit=handoff,
            inputs_sha256={name: {key: digest(root / key) for key in ("predictions.json", "annotations.json", "features.npz")}
                           for name, root in (("wildtrack", args.wildtrack), ("icsens", args.icsens))},
            policy_sha256=digest(args.policy), events=events))
    print(args.output)


if __name__ == "__main__":
    main()
