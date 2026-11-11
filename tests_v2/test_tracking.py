import importlib.util
import unittest
from selfwatch_v2.contracts import FrameRef, Instance
from selfwatch_v2.tracking import LocalTracker


def frame(index, t=None, camera="A", epoch="r1"):
    return FrameRef(camera, epoch, index, float(index if t is None else t), 1000, 640, 480, "test")


def person(x):
    return Instance((x, 10, x+40, 100), .9, 0, "person")


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Requires scipy")
class LocalTrackingTests(unittest.TestCase):
    def test_motion_reordered_detections_and_unique_assignments(self):
        tracker = LocalTracker()
        initial = tracker.update(frame(0), [person(10), person(200)])
        second = tracker.update(frame(1), [person(205), person(15)])
        self.assertEqual([t['local_id'] for t in initial], [1, 2])
        self.assertEqual([t['local_id'] for t in second], [2, 1])
        self.assertEqual(len({t['local_id'] for t in second}), 2)

    def test_no_predicted_people_rendered_and_recovery_before_expiry(self):
        tracker = LocalTracker(max_gap_s=2)
        tracker.update(frame(0), [person(10)])
        self.assertEqual(tracker.update(frame(1), []), [])
        self.assertEqual(tracker.update(frame(2), [person(12)])[0]['local_id'], 1)
        self.assertEqual(tracker.update(frame(3, t=5), [person(12)])[0]['local_id'], 2)

    def test_epoch_resets_but_camera_mixing_is_error(self):
        tracker = LocalTracker()
        tracker.update(frame(0), [person(10)])
        new = tracker.update(frame(0, epoch="reconnect"), [person(10)])[0]
        self.assertEqual(new['stream_epoch'], "reconnect")
        with self.assertRaises(ValueError):
            tracker.update(frame(1, camera="B"), [person(10)])

    def test_actual_label_drives_tracking_not_custom_class_zero(self):
        tracker = LocalTracker()
        self.assertEqual(tracker.update(frame(0), [Instance((0,0,10,10),.9,0,"head")]), [])

    def test_timestamp_monotonicity(self):
        tracker = LocalTracker()
        tracker.update(frame(1), [person(10)])
        with self.assertRaises(ValueError):
            tracker.update(frame(2,t=.5), [person(10)])
