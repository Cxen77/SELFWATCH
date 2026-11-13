"""Freeze and evaluate causal cross-camera links on fixed WILDTRACK inputs."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.download_wildtrack_pilot import IDENTITY_HOLDOUT
from scripts.validate_phase3 import REFERENCE_FLOOR_ID, reference_cameras
from selfwatch_v2.camera_geometry import CameraRig
from selfwatch_v2.contracts import FrameRef
from selfwatch_v2.evaluate_tracking import evaluate
from selfwatch_v2.identity_study import digest
from selfwatch_v2.overlap_memory import LinkPolicy, OverlapMemory, Sighting
from selfwatch_v2.wildtrack_geometry import (
    DEVELOPMENT, EVALUATION, load_cameras, load_run, prepare as prepare_old, run_policy as old_policy,
    summary,
)
from selfwatch_v2.wildtrack_tracking import REFERENCE, pair_scores


GATE = dict(min_pair_precision=.95, min_pair_coverage=.35, min_link_precision=.95,
            min_verified_links=10, min_handoff_opportunities=10, min_handoff_coverage=.50,
            min_returns=10, min_return_coverage=.65)
WINDOW_S = 2.
SOURCE_FILES = ("scripts/validate_phase4.py", "scripts/validate_phase3.py",
                "selfwatch_v2/camera_geometry.py", "selfwatch_v2/overlap_memory.py",
                "selfwatch_v2/evaluate_tracking.py", "selfwatch_v2/wildtrack_geometry.py",
                "selfwatch_v2/wildtrack_tracking.py")


def geometry_inputs(data, cameras, indices, vectors):
    rig = CameraRig(REFERENCE_FLOOR_ID, list(reference_cameras(cameras, indices, "wildtrack-pilot")))
    bundles = defaultdict(list)
    for row in sorted(data["frames"], key=lambda row: (row["frame"]["captured_at_s"], row["frame"]["camera_id"])):
        frame = FrameRef(**row["frame"])
        camera = rig.cameras[frame.camera_id]
        if (frame.width, frame.height) != (camera.floor.width, camera.floor.height):
            raise ValueError("Input differs from calibrated frame geometry")
        sightings = []
        for track in row["tracks_by_case"]["camera_local"]:
            if track.get("status", "observed") != "observed" or track.get("visual_suppressed"):
                continue
            key = (frame.camera_id, frame.stream_epoch, track["local_id"])
            instance = str(track["instance_index"])
            x1, y1, x2, y2 = row["instances"][int(instance)]["box_xyxy"]
            # Box bottom is a noisy proxy. Refuse clipped boxes and increase
            # the pixel envelope with the visible size of the box.
            contact = None
            if 1 < x1 < x2 < frame.width - 1 and 1 < y1 < y2 < frame.height - 1:
                contact = rig.observe(frame, ((x1 + x2) / 2, y2), max(3., .025 * (y2 - y1)))
            feature_key = row["feature_keys"].get(instance)
            feature = None
            if feature_key is not None:
                feature = np.array(vectors[feature_key], dtype=float)
                norm = np.linalg.norm(feature)
                if feature.ndim != 1 or not np.isfinite(norm) or norm <= 1e-8:
                    raise ValueError("Invalid frozen ReID feature")
                feature /= norm
            sightings.append((track, Sighting(key, contact, feature)))
        bundles[frame.captured_at_s].append((row, sightings))
    return rig, sorted(bundles.items())


def replay(data, rig, bundles, policy, *, uncertainty=True, revise=True):
    memory = OverlapMemory(rig, policy, use_uncertainty=uncertainty, revise=revise)
    frames, events = [], []
    counts = Counter()
    update_seconds = 0.
    for timestamp, views in bundles:
        sightings = [obs for _, rows in views for _, obs in rows]
        start = perf_counter()
        ids, changes, alternatives = memory.update(timestamp, sightings)
        update_seconds += perf_counter() - start
        events.extend(changes)
        counts.update(sightings=len(sightings), projected=sum(obs.floor is not None for obs in sightings),
                      geometry_alternatives=len(alternatives),
                      ambiguous_alternatives=sum(not row["separated"] for row in alternatives))
        for row, rows in views:
            tracks = [dict(track, global_id=ids[obs.key]) for track, obs in rows]
            frames.append(dict(row, tracks_by_case=dict(row["tracks_by_case"], overlap=tracks)))
    predictions = dict(data, frames=frames, tracking_cases=dict(data["tracking_cases"], overlap="Revisable overlap"))
    return predictions, events, dict(counts, mean_link_update_ms=1000 * update_seconds / len(bundles))


def opportunities(labels, traces):
    """Measure label-defined view transitions and returns, with misses charged."""
    visible, matched = defaultdict(lambda: defaultdict(set)), defaultdict(dict)
    timeline = sorted({frame["frame"]["captured_at_s"] for frame in labels["frames"]})
    for frame in labels["frames"]:
        t, camera = frame["frame"]["captured_at_s"], frame["frame"]["camera_id"]
        for mark in frame["marks"]:
            visible[mark["subject"]][t].add(camera)
    for trace in traces:
        for row in trace["matches"]:
            matched[row["subject"]][trace["captured_at_s"], trace["camera_id"]] = row["global_id"]

    handoffs, returns = [], []
    for subject, history in visible.items():
        seen, previous_views, previous_time = {}, set(), None
        observed = matched[subject]
        for now, views in sorted(history.items()):
            for camera in views - previous_views:
                if previous_time is not None and now - previous_time <= WINDOW_S:
                    source_views = previous_views - views
                    # A disappearing source and appearing destination with no
                    # sampled overlap. The reference may miss sub-frame overlap.
                    if source_views and not (previous_views & views):
                        anchors = [observed.get((previous_time, source)) for source in source_views]
                        anchors = {anchor for anchor in anchors if anchor is not None}
                        anchor = next(iter(anchors)) if len(anchors) == 1 else None
                        handoffs.append((history, observed, now, camera, anchor))
                prior = seen.get(camera)
                if prior is not None and .5 < now - prior <= WINDOW_S:
                    returns.append((history, observed, now, camera, observed.get((prior, camera))))
            seen.update({camera: now for camera in views})
            previous_views, previous_time = views, now

    def measure(rows):
        delays = []
        available = destination_matched = 0
        for history, observed, start, camera, anchor in rows:
            available += anchor is not None
            delivered = False
            for now in (t for t in timeline if start <= t <= start + WINDOW_S):
                if camera not in history.get(now, ()):
                    break
                got = observed.get((now, camera))
                delivered |= got is not None
                if anchor is not None and anchor == got:
                    delays.append(now - start)
                    break
            destination_matched += delivered
        return dict(opportunities=len(rows), anchor_available=available,
                    destination_matched=destination_matched, recovered=len(delays),
                    coverage=len(delays) / len(rows) if rows else None,
                    median_delay_s=float(np.median(delays)) if delays else None,
                    p95_delay_s=float(np.percentile(delays, 95)) if delays else None)

    return dict(nonoverlap_view_transitions=measure(handoffs), same_view_returns=measure(returns))


def score(data, labels, case, events=()):
    result = evaluate(data, labels, case=case, reference_source=REFERENCE)
    traces = result["frame_matches"]
    by_time = defaultdict(dict)
    frames = {row["observation_id"]: row["frame"] for row in data["frames"]}
    for trace in traces:
        frame = frames[trace["observation_id"]]
        for row in trace["matches"]:
            by_time[trace["captured_at_s"]][frame["camera_id"], frame["stream_epoch"], row["local_id"]] = row["subject"]
    audit = Counter(correct=0, wrong=0, unverified=0, revoked=0)
    for event in events:
        if event["event"] == "revoke":
            audit["revoked"] += 1
            continue
        a, b = (by_time[event["time_s"]].get(key) for key in event["pair"])
        audit["unverified" if a is None or b is None else "correct" if a == b else "wrong"] += 1
    out = summary(dict(result, **pair_scores(labels["frames"], traces)))
    pairs = out["same_person_shared_id"]
    verified = audit["correct"] + audit["wrong"]
    return dict(out, sampled_global_id_changes=result["sampled_global_id_changes"],
                sampled_local_id_changes=result["sampled_local_id_changes"],
                pair_coverage=pairs / max(1, out["same_person_both_matched"]),
                pair_recall_all=pairs / max(1, out["same_person_possible"]),
                link_events=dict(audit, precision=audit["correct"] / verified if verified else None),
                **opportunities(labels, traces))


def passing(result, baseline):
    links, handoffs, returns = (result[key] for key in
                                ("link_events", "nonoverlap_view_transitions", "same_view_returns"))
    return dict(
        pair_precision=(result["matched_pair_precision"] or 0) >= GATE["min_pair_precision"],
        pair_coverage=result["pair_coverage"] >= GATE["min_pair_coverage"],
        link_precision=(links["precision"] or 0) >= GATE["min_link_precision"],
        verified_links=links["correct"] + links["wrong"] >= GATE["min_verified_links"],
        better_IDF1=result["IDF1"] > baseline["IDF1"],
        handoff_support=handoffs["opportunities"] >= GATE["min_handoff_opportunities"],
        handoff_coverage=(handoffs["coverage"] or 0) >= GATE["min_handoff_coverage"],
        return_support=returns["opportunities"] >= GATE["min_returns"],
        return_coverage=(returns["coverage"] or 0) >= GATE["min_return_coverage"],
    )


def frozen_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(mode, run_dir, calibration, policy_path, output):
    indices = {"fit": DEVELOPMENT, "diagnostic": EVALUATION, "evaluate": IDENTITY_HOLDOUT}[mode]
    data, labels, original = load_run(run_dir, indices)
    cameras, calibration_hash = load_cameras(calibration)
    root = Path(__file__).resolve().parents[1]
    sources = {name: sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
               for name in SOURCE_FILES}
    provenance = dict(calibration_sha256=calibration_hash, code_sha256=sources,
                      detector_sha256=original["detector"]["sha256"], reid_sha256=original["reid"]["sha256"])
    with np.load(run_dir / "features.npz", allow_pickle=False) as vectors:
        rig, bundles = geometry_inputs(data, cameras, indices, vectors)
        old_inputs = prepare_old(data, cameras, vectors)
    baseline = score(data, labels, "camera_local")
    if mode == "fit":
        grid = []
        for radius, margin, cosine, weight in product((.9, 1.2, 1.5), (.04, .12), (.45, .55, .65), (1.25, 2.5)):
            policy = LinkPolicy(radius_m=radius, margin_m=margin, min_cosine=cosine,
                                appearance_weight_m=weight)
            predictions, events, _ = replay(data, rig, bundles, policy)
            row = score(predictions, labels, "overlap", events)
            grid.append(dict(policy=asdict(policy), score=row))
        candidates = [row for row in grid if
                      (row["score"]["matched_pair_precision"] or 0) >= GATE["min_pair_precision"]
                      and (row["score"]["link_events"]["precision"] or 0) >= GATE["min_link_precision"]]
        if not candidates:
            raise ValueError("No development setting passed the precision floors")
        chosen = max(candidates, key=lambda row: (
            row["score"]["pair_coverage"], row["score"]["IDF1"], -row["score"]["sampled_global_id_changes"]))
        frozen_write(policy_path, dict(provenance=provenance, gate=GATE, selected=chosen,
                                       development_predictions_sha256=original["predictions_sha256"]))
        frozen_write(output, dict(baseline=baseline, grid=grid))
        return

    freeze = json.loads(policy_path.read_text(encoding="utf-8"))
    if freeze["provenance"] != provenance or freeze["gate"] != GATE:
        raise ValueError("Frozen settings no longer match the source, calibration, or gate")
    policy = LinkPolicy(**freeze["selected"]["policy"])
    arms, evidence = {"camera_local": baseline, "existing_shared_memory": score(data, labels, "legacy")}, {}
    _, _, old = old_policy(data, labels, old_inputs, dict(radius=.75, margin=.25, min_cosine=.55))
    arms["existing_floor_appearance"] = score(old, labels, "geometry")
    for name, uncertainty, revise, settings in (
        ("immediate_point", False, False, LinkPolicy(**dict(asdict(policy), confirmations=1))),
        ("confirmed_point", False, False, policy),
        ("confirmed_interval", True, False, policy),
        ("revisable_interval", True, True, policy),
    ):
        predictions, events, counts = replay(data, rig, bundles, settings,
                                             uncertainty=uncertainty, revise=revise)
        arms[name] = score(predictions, labels, "overlap", events)
        evidence[name] = dict(counts=counts, events=events)
    gate = passing(arms["revisable_interval"], baseline)
    frozen_write(output, dict(
        scope="Temporal outdoor holdout within one recording; box-bottom proxies and published projected labels.",
        mode=mode, frames=len(data["frames"]), dataset_manifest_sha256=original["dataset_manifest_sha256"],
        predictions_sha256=original["predictions_sha256"], annotations_sha256=original["annotations_sha256"],
        features_sha256=original["features_sha256"], policy_sha256=digest(policy_path),
        arms=arms, evidence=evidence, gate=gate, passed=all(gate.values()),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "diagnostic", "evaluate"))
    for name in ("run", "calibration", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.run, args.calibration, args.policy, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
