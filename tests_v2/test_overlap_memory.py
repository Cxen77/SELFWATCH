"""Regressions for causal overlap links and identity evaluation."""
from dataclasses import replace
import unittest

import numpy as np

from scripts.validate_phase4 import opportunities
from selfwatch_v2.camera_geometry import AlignedTime, FloorContact, FloorObservation
from selfwatch_v2.overlap_memory import LinkPolicy, OverlapMemory, Sighting


class TimingRig:
    def shares_coverage(self, a, b, tolerance_s):
        return (a is not None and b is not None and a.floor_id == b.floor_id
                and abs(a.time.time_s - b.time.time_s)
                + a.time.uncertainty_s + b.time.uncertainty_s <= tolerance_s)


def observation(camera, t, x, *, local=1, epoch="first", radius=.03, feature=(1., 0.)):
    point = FloorObservation(camera, epoch, "floor", AlignedTime(t, .025), FloorContact((x, 0.), radius))
    return Sighting((camera, epoch, local), point, np.array(feature) if feature is not None else None)


class OverlapMemoryTests(unittest.TestCase):
    def test_link_and_revoke_without_rewriting_outputs(self):
        memory = OverlapMemory(TimingRig())
        a, b = observation("a", 0, 0), observation("b", 0, .1)
        first, events, _ = memory.update(0, [a, b])
        self.assertFalse(events)
        self.assertNotEqual(first[a.key], first[b.key])
        linked, events, _ = memory.update(.5, [observation("a", .5, 0), observation("b", .5, .1)])
        self.assertEqual(linked[a.key], linked[b.key])
        self.assertEqual(events[0]["event"], "link")
        memory.update(1, [observation("a", 1, 0), observation("b", 1, 2)])
        split, events, _ = memory.update(1.5, [observation("a", 1.5, 0), observation("b", 1.5, 2)])
        self.assertEqual(events[0]["reason"], "contradictory_geometry")
        self.assertEqual(split, first)
        self.assertEqual(linked[a.key], linked[b.key])

    def test_close_candidate_with_missing_feature_blocks_guess(self):
        memory = OverlapMemory(TimingRig(), LinkPolicy(confirmations=1))
        a = observation("a", 0, 0)
        b = observation("b", 0, .05)
        rival = observation("b", 0, .075, local=2, feature=None)
        ids, events, candidates = memory.update(0, [a, b, rival])
        self.assertEqual(len(ids), 3)
        self.assertFalse(events)
        self.assertTrue(all(not row["separated"] for row in candidates))

    def test_clock_and_missing_geometry_withhold_link(self):
        memory = OverlapMemory(TimingRig(), LinkPolicy(confirmations=1))
        a, b = observation("a", 0, 0), observation("b", .2, 0)
        _, events, _ = memory.update(.2, [a, b])
        self.assertFalse(events)
        _, events, _ = memory.update(.3, [observation("a", .3, 0), replace(b, floor=None)])
        self.assertFalse(events)

    def test_expiry_and_new_epoch_do_not_reuse_id(self):
        memory = OverlapMemory(TimingRig(), LinkPolicy(confirmations=1))
        a, b = observation("a", 0, 0), observation("b", 0, 0)
        first, _, _ = memory.update(0, [a, b])
        rejoined = observation("b", .5, 2, epoch="second")
        second, events, _ = memory.update(.5, [rejoined])
        self.assertEqual(events[0]["reason"], "owner_expired")
        self.assertNotEqual(first[b.key], second[rejoined.key])
        back, _, _ = memory.update(6, [observation("a", 6, 0)])
        self.assertNotEqual(first[a.key], back[a.key])

    def test_component_cannot_alias_two_owners_in_one_view(self):
        memory = OverlapMemory(TimingRig(), LinkPolicy(confirmations=1, radius_m=1.1))
        a, b = observation("a", 0, 0), observation("b", 0, .1)
        ids, _, _ = memory.update(0, [a, b])
        self.assertEqual(ids[a.key], ids[b.key])
        other = observation("a", .5, .1, local=2)
        ids, events, _ = memory.update(.5, [other, observation("b", .5, .1)])
        self.assertFalse(events)
        self.assertNotEqual(ids[other.key], ids[b.key])

    def test_transition_denominator_counts_missing_anchor_and_late_return(self):
        labels, traces = {"frames": []}, []
        rows = [(0, "a", "P", "1"), (.5, "b", "P", None), (1, "b", "P", "1"),
                (1.5, "a", "P", None), (2, "a", "P", "1")]
        for t, camera, subject, gid in rows:
            labels["frames"].append(dict(frame=dict(captured_at_s=t, camera_id=camera),
                                         marks=[dict(subject=subject)]))
            traces.append(dict(captured_at_s=t, camera_id=camera,
                               matches=[dict(subject=subject, global_id=gid)] if gid else []))
        scores = opportunities(labels, traces)
        self.assertEqual(scores["nonoverlap_view_transitions"]["opportunities"], 2)
        self.assertEqual(scores["nonoverlap_view_transitions"]["recovered"], 2)
        self.assertEqual(scores["same_view_returns"]["opportunities"], 1)
        self.assertEqual(scores["same_view_returns"]["median_delay_s"], .5)


if __name__ == "__main__":
    unittest.main()
