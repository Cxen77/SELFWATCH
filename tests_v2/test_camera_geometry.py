"""Phase 3 regressions for spatial, temporal, and zone evidence."""
from dataclasses import replace
import unittest

import cv2
import numpy as np

from scripts.validate_phase3 import survey_rig
from selfwatch_v2.camera_geometry import CameraClock, FloorMap
from selfwatch_v2.contracts import FrameRef


def survey_fixture():
    data = dict(schema_version=1, floor_id="synthetic-floor-v1", cameras=[],
                layout=dict(zones=[], exit_gates=[]))
    fit_m = [[0, 0], [10, 0], [10, 10], [0, 10]]
    check_m = [[2, 2], [8, 2], [5, 7]]

    for camera_id, offset in (("left", 0.2), ("right", -0.35)):
        def events(seconds):
            return [[t, offset + 1.001 * t] for t in seconds]

        data["cameras"].append(dict(
            camera_id=camera_id, stream_epoch="first", model="fixture", view=camera_id,
            size=[1000, 1000], lens={"input": "undistorted"},
            survey=dict(fit_m=fit_m, fit_px=(np.array(fit_m) * 60 + 120).tolist(),
                        check_m=check_m, check_px=(np.array(check_m) * 60 + 120).tolist(),
                        max_error_m=.05),
            clock=dict(fit_events=events([0, 4, 8]), check_events=events([2, 6]),
                       quantization_s=.01, max_error_s=.03),
        ))
    for x, kind in ((2, "aisle"), (4, "shelf"), (6, "checkout")):
        data["layout"]["zones"].append(dict(name=kind, kind=kind, boundary_m=[
            [x - .4, 4], [x + .4, 4], [x + .4, 6], [x - .4, 6],
        ]))
    data["layout"]["exit_gates"].append(dict(
        name="front", start_m=[7, 6], end_m=[7, 4],
        hysteresis_m=.1, max_gap_s=1.5, max_speed_m_s=3,
    ))
    return data


class CameraGeometryTests(unittest.TestCase):
    def test_raw_lens_fit_holds_out_independent_positions(self):
        camera = np.array([[600., 0, 500], [0, 600, 500], [0, 0, 1]])
        distortion = np.array([.18, .02, .003, -.001, 0])
        fit = np.array([[-2., -2], [2, -2], [2, 2], [-2, 2]])
        check = np.array([[1., 1], [-1.4, -.5], [-.5, 1.5]])

        def distort(points):
            world = np.column_stack((points, np.zeros(len(points))))
            pixels, _ = cv2.projectPoints(world, np.array([.14, -.23, .03]),
                                          np.array([.4, -.3, 5]), camera, distortion)
            return pixels[:, 0]

        args = ((1000, 1000), distort(fit), fit, distort(check), check)
        floor, stats = FloorMap.fit(*args, max_error_m=.01, pinhole=camera, distortion=distortion)
        self.assertEqual(stats.count, 3)
        self.assertLess(stats.maximum, .001)
        contact = floor.contact(distort([[0., 0]])[0], 3, (1000, 1000))
        np.testing.assert_allclose(contact.xy_m, [0, 0], atol=.001)
        self.assertGreater(contact.radius_m, .025)
        self.assertIsNone(floor.contact(distort([[2.2, 0]])[0], 3, (1000, 1000)))
        self.assertIsNone(floor.contact(distort([[0., 0]])[0], 3, (999, 1000)))
        wrong = check + [.1, 0]
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            FloorMap.fit(*args[:-1], wrong, max_error_m=.01, pinhole=camera, distortion=distortion)
        with self.assertRaisesRegex(ValueError, "distinct"):
            FloorMap.fit((1000, 1000), distort(fit), fit, distort(fit[:3]), fit[:3],
                         max_error_m=.01)

    def test_clock_rejects_drift_beyond_held_out_gate(self):
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            CameraClock.fit([[0, .1], [4, 4.1], [8, 8.1]], [[2, 2.1], [6, 6.3]],
                            quantization_s=.01, max_error_s=.03)

    def test_survey_rig_time_topology_and_exit(self):
        rig, zones, exits, report = survey_rig(survey_fixture())
        self.assertEqual(report["overlap_pairs"], [("left", "right")])
        self.assertAlmostEqual(report["cameras"]["left"]["offset_at_origin_s"], .204)
        self.assertAlmostEqual(report["cameras"]["right"]["clock"]["rate"], 1.001)
        frame = FrameRef("left", "first", 1, 4, 4.3, 1000, 1000, "fixture")

        def at(x, y, ref=frame):
            return rig.observe(ref, (120 + 60 * x, 120 + 60 * y), 2)

        left = at(6, 5)
        right_ref = replace(frame, camera_id="right", captured_at_s=4 + .55 / 1.001)
        right = at(6, 5, right_ref)
        self.assertTrue(rig.shares_coverage(left, right, .03))
        self.assertFalse(rig.shares_coverage(left, right, .015))
        self.assertFalse(rig.shares_coverage(left, at(6, 5, replace(right_ref, captured_at_s=5)), .03))
        self.assertFalse(rig.shares_coverage(left, None, .03))
        self.assertIsNone(at(6, 5, replace(frame, camera_id="missing")))
        self.assertIsNone(at(6, 5, replace(frame, stream_epoch="reset")))
        self.assertIsNone(at(6, 5, replace(frame, captured_at_s=9)))
        self.assertTrue(zones["checkout"].contains(left))
        self.assertFalse(zones["checkout"].contains(at(6.39, 5)))
        gate = exits["front"]
        later = replace(frame, frame_index=2, captured_at_s=4.8)
        self.assertTrue(gate.crossed_outbound(at(6.5, 5), at(7.5, 5, later)))
        self.assertFalse(gate.crossed_outbound(at(7.5, 5), at(6.5, 5, later)))
        self.assertFalse(gate.crossed_outbound(at(6.5, 6.5), at(7.5, 6.5, later)))
        self.assertFalse(gate.crossed_outbound(at(6.95, 5), at(7.5, 5, later)))
        self.assertFalse(gate.crossed_outbound(at(6.5, 5), at(7.5, 5, replace(later, captured_at_s=6))))
        self.assertFalse(gate.crossed_outbound(at(6.5, 5), at(7.5, 5, replace(later, captured_at_s=4.2))))

    def test_survey_refuses_uncovered_zone(self):
        data = survey_fixture()
        data["layout"]["zones"][0]["boundary_m"][0] = [-10, -10]
        with self.assertRaisesRegex(ValueError, "outside calibrated coverage"):
            survey_rig(data)


if __name__ == "__main__":
    unittest.main()
