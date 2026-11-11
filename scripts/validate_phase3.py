"""Validate a surveyed rig or reproduce the scoped WILDTRACK geometry report."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np

from selfwatch_v2.camera_geometry import (
    CameraCalibration, CameraClock, CameraRig, ErrorStats, ExitGate, FloorMap, Zone,
)
from selfwatch_v2.contracts import FrameRef
from selfwatch_v2.identity_study import digest
from selfwatch_v2.wildtrack_geometry import ARCHIVE, EVALUATION, load_cameras
from selfwatch_v2.wildtrack_tracking import marks_for_camera


REFERENCE_FLOOR = ((-3, -9), (9, -9), (9, 27), (-3, 27))
REFERENCE_EPOCH = "wildtrack-evaluation-v1"
REFERENCE_FLOOR_ID = "wildtrack-plaza-v1"
PAIR_TOLERANCE_S = 0.15


def survey_rig(data):
    """Build the same rig contract from site measurements and return its audit."""
    if data["schema_version"] != 1:
        raise ValueError("Unsupported survey schema")
    floor_id = data["floor_id"]
    cameras, errors = [], {}
    for item in data["cameras"]:
        camera_id, epoch = item["camera_id"], item["stream_epoch"]
        if not camera_id or not epoch or not item["model"] or not item["view"]:
            raise ValueError("Camera ID, epoch, model and physical view are required")
        lens = item["lens"]
        if lens["input"] == "raw":
            lens_options = dict(pinhole=lens["pinhole"], distortion=lens["distortion"])
        elif lens["input"] == "undistorted":
            lens_options = {}
        else:
            raise ValueError("Declare raw pinhole or already undistorted input")
        targets, timing = item["survey"], item["clock"]
        floor, spatial = FloorMap.fit(
            item["size"], targets["fit_px"], targets["fit_m"],
            targets["check_px"], targets["check_m"],
            max_error_m=targets["max_error_m"], **lens_options,
        )
        clock, temporal = CameraClock.fit(
            timing["fit_events"], timing["check_events"],
            quantization_s=timing["quantization_s"], max_error_s=timing["max_error_s"],
        )
        cameras.append(CameraCalibration(camera_id, epoch, floor_id, floor, clock))
        errors[camera_id] = dict(
            model=item["model"], view=item["view"], stream_epoch=epoch,
            size=item["size"], floor_residual_m=asdict(spatial), clock_residual_s=asdict(temporal),
            clock=asdict(clock), offset_at_origin_s=clock.rig_origin_s - clock.raw_origin_s,
            floor_from_undistorted_pixel=floor.matrix.tolist(),
            validated_footprint_m=floor.footprint.tolist(),
        )
    rig = CameraRig(floor_id, cameras)
    layout = data["layout"]
    zones, exits = _layout(rig, layout)
    return rig, zones, exits, dict(
        floor_id=floor_id, cameras=errors, overlap_pairs=sorted(rig.overlaps), layout=layout,
    )


def _layout(rig, layout):
    zones = {}
    exits = {}
    for item in layout["zones"]:
        name = item["name"]
        if not name or name in zones:
            raise ValueError("Zone names must be unique and nonempty")
        zones[name] = Zone(rig.floor_id, item["kind"], item["boundary_m"])
    if {zone.kind for zone in zones.values()} != {"aisle", "shelf", "checkout"}:
        raise ValueError("Layout must define aisle, shelf and checkout zones")
    for item in layout["exit_gates"]:
        name = item["name"]
        if not name or name in exits:
            raise ValueError("Exit gate names must be unique and nonempty")
        exits[name] = ExitGate(rig.floor_id, item["start_m"], item["end_m"],
                               hysteresis_m=item["hysteresis_m"], max_gap_s=item["max_gap_s"],
                               max_speed_m_s=item["max_speed_m_s"])
    if not exits:
        raise ValueError("Layout must define at least one directional exit gate")
    areas = [zone.boundary for zone in zones.values()]
    areas.extend([item["start_m"], item["end_m"]] for item in layout["exit_gates"])
    for points in areas:
        if not any(all(cv2.pointPolygonTest(camera.floor.footprint, tuple(map(float, p)), True)
                       > camera.floor.residual_m for p in points) for camera in rig.cameras.values()):
            raise ValueError("A zone or exit extends outside calibrated coverage")
    return zones, exits


def _clip_polygon(vertices, line):
    """Clip a floor polygon by one linear inequality."""
    result = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        a, b = np.dot(line, [*start, 1]), np.dot(line, [*end, 1])
        if a >= 0:
            result.append(start)
        if (a < 0) != (b < 0):
            result.append(start + a / (a - b) * (end - start))
    return np.asarray(result)


def _reference_cameras(cameras):
    clock = CameraClock(0, 0, 1, EVALUATION[0] / 10, EVALUATION[-1] / 10, .05)
    for name, camera in cameras.items():
        image = np.linalg.inv(camera.floor_from_image)
        x, y, depth = image
        footprint = np.array(REFERENCE_FLOOR, dtype=float)
        for plane in (depth, x, y, (camera.width - 1) * depth - x,
                      (camera.height - 1) * depth - y):
            footprint = _clip_polygon(footprint, plane)
        floor = FloorMap((camera.width, camera.height), camera.floor_from_image,
                         footprint, residual_m=0)
        yield CameraCalibration(name, REFERENCE_EPOCH, REFERENCE_FLOOR_ID, floor, clock)


def _reference_frame(camera, index):
    return FrameRef(camera.camera_id, camera.stream_epoch, index, index / 10, index / 10,
                    camera.floor.width, camera.floor.height, "official:wildtrack",
                    "published synchronized frame index / 10")


def _diagnose_feet(rig, marks, labels):
    positions = {row["personID"]: row["positionID"] for row in labels}
    index = marks["frame_index"]
    distances, by_subject, rows = defaultdict(list), defaultdict(list), []
    seen = set()
    for mark in marks["marks"]:
        name, subject = mark["camera"], mark["subject"]
        camera = rig.cameras[name]
        if (name, subject) in seen or str(subject) not in {
            row["subject"] for row in marks_for_camera(labels, int(name[1:]), 1920, 1080)
        }:
            raise ValueError(f"Invalid marked subject {name}/{subject}")
        seen.add((name, subject))
        obs = rig.observe(_reference_frame(camera, index), mark["xy_px"], marks["pixel_error"])
        if obs is None:
            rows.append(dict(mark, status="withheld"))
            continue
        grid = positions[subject]
        if not 0 <= grid < 480 * 1440:
            raise ValueError("Ground-grid position outside published grid")
        target = (-3 + .025 * (grid % 480), -9 + .025 * (grid // 480))
        error = float(np.linalg.norm(np.asarray(obs.contact.xy_m) - target))
        distances[name].append(error)
        by_subject[subject].append(obs)
        rows.append(dict(mark, status="projected", published_xy_m=target,
                         error_m=error, contact=asdict(obs.contact)))
    if any(len(distances[name]) < 4 for name in rig.cameras):
        raise ValueError("Need at least four usable marked contacts per camera")
    gaps = [
        float(np.linalg.norm(np.asarray(a.contact.xy_m) - b.contact.xy_m))
        for observations in by_subject.values()
        for a, b in combinations(observations, 2)
        if rig.shares_coverage(a, b, PAIR_TOLERANCE_S)
    ]
    if not gaps:
        raise ValueError("Need independently marked contacts in shared camera coverage")
    return dict(
        against_published_m={name: asdict(ErrorStats.measure(values))
                             for name, values in sorted(distances.items())},
        same_subject_cross_view_m=asdict(ErrorStats.measure(gaps)), marks=rows,
    )


def _layout_exercise(rig):
    """Use simulated zone positions on the published map, separate from people."""
    camera = rig.cameras["C1"]
    x, y = map(float, camera.floor.footprint.mean(axis=0))
    zones = []
    for name, cx in (("aisle", x - .8), ("shelf", x), ("checkout", x + .8)):
        zones.append(dict(name=name, kind=name, boundary_m=[
            [cx - .3, y - .35], [cx + .3, y - .35],
            [cx + .3, y + .35], [cx - .3, y + .35],
        ]))
    gate = dict(name="exit", start_m=[x + 1.2, y + .5], end_m=[x + 1.2, y - .5],
                hysteresis_m=.08, max_gap_s=1, max_speed_m_s=3)
    layout = dict(zones=zones, exit_gates=[gate])
    compiled, gates = _layout(rig, layout)
    image = np.linalg.inv(camera.floor.matrix)

    def observe(px, py, index):
        point = image @ [px, py, 1]
        frame = _reference_frame(camera, index)
        return rig.observe(frame, point[:2] / point[2], 2)

    center = observe(x + .8, y, 1300)
    before, after = observe(x + .9, y, 1300), observe(x + 1.5, y, 1305)
    inbound_before = observe(x + 1.5, y, 1300)
    inbound_after = observe(x + .9, y, 1305)
    gate_test = gates["exit"]
    frame = _reference_frame(camera, 1300)
    checks = dict(
        checkout=compiled["checkout"].contains(center),
        other_zones_excluded=not any(compiled[name].contains(center) for name in ("aisle", "shelf")),
        boundary_withheld=not compiled["checkout"].contains(observe(x + .8, y + .35, 1300)),
        outbound=gate_test.crossed_outbound(before, after),
        inbound_withheld=not gate_test.crossed_outbound(inbound_before, inbound_after),
        missing_exit_contact_withheld=not gate_test.crossed_outbound(None, after),
        clipped_contact_withheld=rig.observe(frame, (1, 100), 2) is None,
        missing_camera_withheld=rig.observe(replace(frame, camera_id="offline"), (500, 500), 2) is None,
        reset_clock_withheld=rig.observe(replace(frame, stream_epoch="reset"), (500, 500), 2) is None,
        stale_clock_withheld=rig.observe(_reference_frame(camera, 1295), (500, 500), 2) is None,
    )
    if not all(checks.values()):
        raise ValueError(f"Reference layout exercise failed: {checks}")
    return dict(layout=layout, checks=checks)


def reference_report(dataset, marks_path):
    cameras, calibration_hash = load_cameras(dataset)
    rig = CameraRig(REFERENCE_FLOOR_ID, list(_reference_cameras(cameras)))
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["etag"] != ARCHIVE[1] or tuple(manifest["frames"]) != EVALUATION:
        raise ValueError("Expected the frozen WILDTRACK evaluation segment")
    records = {row["path"]: row for row in manifest["files"]}

    def verify(relative):
        path = dataset / relative
        record = records[relative]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Changed dataset file: {relative}")
        return path

    for index in EVALUATION:
        for name in rig.cameras:
            verify(f"Image_subsets/{name}/{index:08d}.png")
    marks = json.loads(marks_path.read_text(encoding="utf-8"))
    if marks["schema_version"] != 1 or marks["frame_index"] not in EVALUATION:
        raise ValueError("Invalid reference marks")
    labels_path = verify(f"annotations_positions/{marks['frame_index']:08d}.json")
    diagnostic = _diagnose_feet(rig, marks, json.loads(labels_path.read_text()))
    return dict(
        scope="Public outdoor WILDTRACK reference rig, separate from store commissioning.",
        dataset_manifest_sha256=digest(manifest_path), calibration_manifest_sha256=calibration_hash,
        manual_marks_sha256=digest(marks_path), camera_count=len(rig.cameras),
        overlap_pairs=sorted(rig.overlaps), frame_bundles=len(EVALUATION),
        verified_frames=len(EVALUATION) * len(rig.cameras),
        timestamp_basis="Published synchronized frame index / 10; no measured per-camera offsets.",
        camera_clock_uncertainty_assumed_s=.05, pair_tolerance_s=PAIR_TOLERANCE_S,
        geometry_scope="Published optics/extrinsics. Manual feet compared to supplied ground grid; "
                       "no surveyed ground truth. Contact radii here represent pixel sensitivity only.",
        cameras={name: dict(size=[cam.floor.width, cam.floor.height],
                            floor_from_undistorted_pixel=cam.floor.matrix.tolist(),
                            visible_footprint_m=cam.floor.footprint.tolist())
                 for name, cam in rig.cameras.items()},
        visible_feet=diagnostic, simulated_layout=_layout_exercise(rig),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("dataset", type=Path)
    reference.add_argument("marks", type=Path)
    reference.add_argument("report", type=Path)
    survey = commands.add_parser("survey")
    survey.add_argument("config", type=Path)
    survey.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.mode == "reference":
        report = reference_report(args.dataset, args.marks)
    else:
        _, _, _, report = survey_rig(json.loads(args.config.read_text(encoding="utf-8")))
        report["survey_sha256"] = digest(args.config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
