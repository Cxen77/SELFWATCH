"""
Tests for visible ID switch detection metrics.

These tests verify that the evaluation pipeline correctly detects
what a human viewer would see as identity switches on screen.
"""

import pytest
from evaluation.visual_metrics.visible_id_switches import VisibleIDSwitchMetric
from evaluation.visual_metrics.box_continuity import BoxContinuityMetric, box_iou


def make_visible_object(gid, x1, y1, x2, y2):
    return {'global_id': gid, 'bbox': [x1, y1, x2, y2]}


# ── VisibleIDSwitchMetric Tests ────────────────────────────────────────

class TestVisibleIDSwitchTypeA:
    """TYPE-A: Vanish + Replace detection"""

    def test_basic_replace(self):
        """gid=5 visible for 10 frames → disappears → gid=12 appears at same position"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10)

        # Establish gid=5
        for f in range(10):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        assert metric._visibility_duration[5] == 10

        # Gap: gid=5 disappears for 3 frames
        metric.update(10, [])
        metric.update(11, [])
        metric.update(12, [make_visible_object(12, 102, 101, 202, 301)])

        metrics = metric.get_metrics()
        assert metrics['replace_switches'] == 1
        assert metrics['visible_id_switches'] >= 1

    def test_replace_too_short_visibility(self):
        """gid visible for only 3 frames (below min) → no switch"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10)

        for f in range(3):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        metric.update(3, [make_visible_object(12, 102, 101, 202, 301)])

        metrics = metric.get_metrics()
        assert metrics['visible_id_switches'] == 0

    def test_replace_outside_spatial_tolerance(self):
        """Appearance is too far from disappearance point → no switch"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10)

        for f in range(10):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        # Appears 200px away
        metric.update(10, [make_visible_object(12, 400, 400, 500, 600)])

        metrics = metric.get_metrics()
        assert metrics['visible_id_switches'] == 0

    def test_replace_outside_temporal_window(self):
        """Gap exceeds temporal_window → no switch"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=3)

        for f in range(10):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        # Wait too long (5 frames gap)
        for f in range(10, 15):
            metric.update(f, [])

        metric.update(15, [make_visible_object(12, 102, 101, 202, 301)])

        metrics = metric.get_metrics()
        assert metrics['visible_id_switches'] == 0

    def test_normal_birth_not_switch(self):
        """Genuinely new person appearing (no predecessor in region) → no switch"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10)

        for f in range(10):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        # New person far from existing track → birth, not switch
        metric.update(10, [
            make_visible_object(5, 100, 100, 200, 300),
            make_visible_object(99, 500, 500, 600, 700),
        ])

        metrics = metric.get_metrics()
        assert metrics['visible_id_switches'] == 0

    def test_flicker_resilience(self):
        """Brief 1-frame flicker should not reset visibility duration"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10,
            flicker_grace_frames=3)

        for f in range(8):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        # Brief 1-frame disappearance
        metric.update(8, [])

        # Reappears at same position
        metric.update(9, [make_visible_object(5, 102, 101, 202, 301)])

        # Duration should have been preserved, not reset
        # After 10 frames active, it should have recovered
        for f in range(10, 15):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        assert metric._visibility_duration.get(5, 0) >= 8


class TestVisibleIDSwitchTypeB:
    """TYPE-B: ID Swap detection"""

    def test_identity_swap(self):
        """Two identities exchange positions"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10,
            min_swap_displacement=50)

        # Establish gid=3 on left, gid=7 on right
        for f in range(10):
            metric.update(f, [
                make_visible_object(3, 100, 200, 200, 400),  # left
                make_visible_object(7, 400, 200, 500, 400),  # right
            ])

        # Swap: gid=3 moves to right, gid=7 moves to left
        metric.update(10, [
            make_visible_object(3, 400, 200, 500, 400),  # was gid=7's position
            make_visible_object(7, 100, 200, 200, 400),  # was gid=3's position
        ])

        metrics = metric.get_metrics()
        assert metrics['swap_switches'] >= 1

    def test_no_swap_on_normal_movement(self):
        """Slow drift is not a swap"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10,
            min_swap_displacement=50)

        for f in range(10):
            metric.update(f, [
                make_visible_object(3, 100, 200, 200, 400),
                make_visible_object(7, 400, 200, 500, 400),
            ])

        # Small movement (not enough for swap detection)
        metric.update(10, [
            make_visible_object(3, 105, 202, 205, 402),  # 5px move
            make_visible_object(7, 395, 198, 495, 398),  # 5px move
        ])

        metrics = metric.get_metrics()
        assert metrics['swap_switches'] == 0


class TestVisibleIDSwitchTypeC:
    """TYPE-C: Flicker / Recovery Failure detection"""

    def test_recovery_failure(self):
        """gid disappears then reappears far from original position"""
        metric = VisibleIDSwitchMetric(
            spatial_tolerance=60, min_visible_frames=5, temporal_window=10,
            flicker_grace_frames=1)

        for f in range(10):
            metric.update(f, [make_visible_object(5, 100, 100, 200, 300)])

        # Gap
        metric.update(10, [])
        metric.update(11, [])

        # Reappears 200px away
        metric.update(12, [make_visible_object(5, 400, 400, 500, 600)])

        metrics = metric.get_metrics()
        assert metrics['flicker_switches'] >= 1


# ── BoxContinuityMetric Tests ──────────────────────────────────────────

class TestBoxContinuity:
    def test_box_tracking_consistent(self):
        """Same gid in same-ish position → continuity maintained"""
        metric = BoxContinuityMetric(iou_threshold=0.20)

        for f in range(5):
            x_offset = f * 2
            metric.update(f, [
                make_visible_object(1, 100 + x_offset, 100, 200 + x_offset, 300),
            ])

        m = metric.get_metrics()
        assert m['box_continuity_switches'] == 0
        assert m['display_id_changes'] == 0

    def test_display_id_change(self):
        """Same spatial position, different gid → display ID change"""
        metric = BoxContinuityMetric(iou_threshold=0.20)

        for f in range(5):
            metric.update(f, [make_visible_object(1, 100, 100, 200, 300)])

        # Same position, different gid
        metric.update(5, [make_visible_object(2, 100, 100, 200, 300)])

        m = metric.get_metrics()
        assert m['display_id_changes'] >= 1

    def test_birth_detection(self):
        """Genuinely new box → birth, not switch"""
        metric = BoxContinuityMetric(iou_threshold=0.20)

        for f in range(5):
            metric.update(f, [make_visible_object(1, 100, 100, 200, 300)])

        # New box in completely different location
        metric.update(5, [
            make_visible_object(1, 100, 100, 200, 300),
            make_visible_object(99, 500, 500, 600, 700),
        ])

        m = metric.get_metrics()
        assert m['box_births'] >= 1
        assert m['box_continuity_switches'] == 0

    def test_box_continuity_switch_near_death(self):
        """Box appears where recently died box was → continuity switch"""
        metric = BoxContinuityMetric(iou_threshold=0.20, max_death_memory=10)

        for f in range(5):
            metric.update(f, [make_visible_object(1, 100, 100, 200, 300)])

        # Box dies
        metric.update(5, [])

        # New box appears at near-exact same position, different gid
        metric.update(6, [make_visible_object(2, 102, 101, 202, 301)])

        m = metric.get_metrics()
        assert m['box_continuity_switches'] >= 1


# ── Box IoU Helper Tests ───────────────────────────────────────────────

class TestBoxIoU:
    def test_perfect_overlap(self):
        assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_partial_overlap(self):
        iou = box_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.1 < iou < 0.2  # ~0.142


# ── Integration Tests ──────────────────────────────────────────────────

class TestEvaluatorIntegration:
    """Tests the full evaluator output format"""

    def test_get_final_report_structure(self):
        from evaluation.evaluator import SELFWATCHEvaluator
        evaluator = SELFWATCHEvaluator()

        for f in range(10):
            evaluator.update(f, [make_visible_object(1, 100, 100, 200, 300)])

        report = evaluator.get_final_report()

        # Verify new fields exist
        assert 'total_visible_switches' in report['summary']
        assert 'switch_breakdown' in report['summary']
        assert 'box_continuity_switches' in report['summary']
        assert 'display_id_changes' in report['summary']
        assert 'replace' in report['summary']['switch_breakdown']
        assert 'swap' in report['summary']['switch_breakdown']
        assert 'flicker' in report['summary']['switch_breakdown']
        assert 'box_continuity' in report['summary']['switch_breakdown']
        assert 'display_id_change' in report['summary']['switch_breakdown']
        assert 'box_continuity' in report['detailed']

    def test_forensic_report(self):
        from evaluation.evaluator import SELFWATCHEvaluator
        evaluator = SELFWATCHEvaluator()

        for f in range(10):
            evaluator.update(f, [make_visible_object(1, 100, 100, 200, 300)])

        # Force a replace switch
        evaluator.visible_id_switches.update(10, [])
        evaluator.visible_id_switches.update(11, [])
        evaluator.visible_id_switches.update(12, [make_visible_object(2, 102, 101, 202, 301)])

        report = evaluator.get_forensic_report()
        assert 'TYPE-A' in report
        assert 'REPLACE' in report
        assert '2' in report

    def test_evaluator_detects_replace_in_runtime(self):
        """Full evaluator should detect basic replace scenario"""
        from evaluation.evaluator import SELFWATCHEvaluator
        evaluator = SELFWATCHEvaluator()

        for f in range(10):
            evaluator.update(f, [make_visible_object(1, 100, 100, 200, 300)])

        # Gap with placeholder empty frames
        evaluator.update(10, [])
        evaluator.update(11, [])
        evaluator.update(12, [make_visible_object(2, 102, 101, 202, 301)])

        report = evaluator.get_final_report()
        assert report['summary']['total_visible_switches'] >= 1
