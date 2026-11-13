"""Regression checks for the Phase 4 continuity candidates."""
from types import SimpleNamespace
import unittest

import numpy as np

from selfwatch_v2.appearance_motion import AppearanceMotion, LocalPolicy
from selfwatch_v2.contracts import Instance
from selfwatch_v2.overlap_memory import Sighting
from selfwatch_v2.transit_memory import TransitMemory, TransitPolicy


def sight(camera, number, feature=(1., 0.)):
    return Sighting((camera, "first", number), feature=np.asarray(feature) if feature else None)


class ContinuityTests(unittest.TestCase):
    def test_local_track_uses_measured_return_and_fresh_epoch(self):
        tracker = AppearanceMotion(LocalPolicy())
        box = Instance((10, 10, 50, 100), .8, 0, "person")
        frame = lambda t, epoch="first": SimpleNamespace(captured_at_s=t, stream_epoch=epoch)
        features = {0: np.asarray([1., 0.])}
        original = tracker.update(frame(0), [box], features)[0]["local_id"]
        self.assertEqual(tracker.update(frame(.5), [], {}), [])
        self.assertEqual(tracker.update(frame(1.), [box], features)[0]["local_id"], original)
        self.assertNotEqual(tracker.update(frame(0., "second"), [box], features)[0]["local_id"], original)

    def test_transit_requires_departure_and_keeps_past_assignments(self):
        tracker = TransitMemory([("a", "b")], TransitPolicy())
        a, b = sight("a", 1), sight("b", 1)
        ids = tracker.update(0., [a, b])
        first = tracker.update(.5, [b])
        later = tracker.update(1., [b])
        self.assertNotEqual(ids[a.key], ids[b.key])
        self.assertEqual(first[b.key], ids[b.key])
        self.assertEqual(later[b.key], ids[a.key])
        reunited = tracker.update(1.5, [a, b])
        self.assertEqual(reunited[a.key], ids[a.key])
        self.assertEqual(reunited[b.key], ids[b.key])

    def test_ambiguous_destinations_and_missing_features_abstain(self):
        tracker = TransitMemory([("a", "b")], TransitPolicy(confirmations=1))
        a, b, c = sight("a", 1), sight("b", 1), sight("b", 2)
        missing = sight("b", 3, feature=None)
        ids = tracker.update(0., [a])
        output = tracker.update(.5, [b, c, missing])
        self.assertEqual(len(set(output.values())), 3)
        self.assertNotIn(ids[a.key], output.values())

    def test_active_owner_and_unconfigured_direction_are_ineligible(self):
        tracker = TransitMemory([("a", "b")], TransitPolicy(confirmations=1))
        a, b = sight("a", 1), sight("b", 1)
        start = tracker.update(0., [b])
        latest = tracker.update(.5, [a])
        self.assertNotEqual(start[b.key], latest[a.key])
        together = tracker.update(1., [a, b])
        self.assertNotEqual(together[a.key], together[b.key])


if __name__ == "__main__":
    unittest.main()
