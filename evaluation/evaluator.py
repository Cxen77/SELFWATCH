from evaluation.visual_metrics.visible_id_switches import VisibleIDSwitchMetric
from evaluation.visual_metrics.box_continuity import BoxContinuityMetric
from evaluation.visual_metrics.duplicate_box import DuplicateBoxMetric
from evaluation.visual_metrics.fragmentation import FragmentationMetric
from evaluation.teleportation_metrics.teleportation import IdentityTeleportationMetric
from evaluation.stability_metrics.occlusion_recovery import OcclusionRecoveryMetric
from evaluation.stability_metrics.identity_stability import IdentityStabilityScore


class SELFWATCHEvaluator:
    """
    Master evaluator that orchestrates all visible continuity and stability metrics.

    Tuned for ~5 FPS human-perceptual measurement:
      - min_visible_frames=5 (1 second, down from 15/3s)
      - temporal_window=10 (2 second gap, up from 2/400ms)
      - spatial_tolerance=60.0 (more forgiving for motion)
    """

    def __init__(self):
        self.visible_id_switches = VisibleIDSwitchMetric(
            spatial_tolerance=60.0, min_visible_frames=5, temporal_window=10,
            min_swap_displacement=50.0, flicker_grace_frames=3)
        self.box_continuity = BoxContinuityMetric(
            iou_threshold=0.20, max_death_memory=10)
        self.duplicate_box = DuplicateBoxMetric()
        self.fragmentation = FragmentationMetric()
        self.teleportation = IdentityTeleportationMetric()
        self.occlusion_recovery = OcclusionRecoveryMetric()
        self.stability_scorer = IdentityStabilityScore()

        self.total_frames = 0
        self.active_tracks_accumulated = 0

    def update(self, frame_idx: int, visible_rendered_identities: list,
               tracks: list = None, detections: list = None,
               suppression_regions: list = None, frozen_gids: list = None,
               raw_display: dict = None):
        """
        Process a single frame's visual output and tracking state.

        raw_display: optional dict of what WOULD have been rendered
                     without ownership arbitration, for diagnostic comparison.
        """
        self.total_frames += 1
        self.active_tracks_accumulated += len(visible_rendered_identities)

        self.visible_id_switches.update(frame_idx, visible_rendered_identities,
                                        raw_display=raw_display)
        self.box_continuity.update(frame_idx, visible_rendered_identities)
        self.duplicate_box.update(frame_idx, visible_rendered_identities)
        self.fragmentation.update(frame_idx, visible_rendered_identities)
        self.teleportation.update(frame_idx, visible_rendered_identities)

    def get_final_report(self):
        switches_data = self.visible_id_switches.get_metrics()
        box_cont_data = self.box_continuity.get_metrics()
        dupes_data = self.duplicate_box.get_metrics()
        teleport_data = self.teleportation.get_metrics()
        frag_data = self.fragmentation.get_metrics()

        avg_active = self.active_tracks_accumulated / max(1, self.total_frames)

        # Total visible switches across all detection modes
        total_visible_switches = (
            switches_data['visible_id_switches'] +
            box_cont_data['box_continuity_switches'] +
            box_cont_data['display_id_changes']
        )

        stability_score = self.stability_scorer.compute_score(
            visible_switches=total_visible_switches,
            duplicates=dupes_data['duplicate_frame_count'],
            teleportations=teleport_data['teleportation_count'],
            fragmentations=frag_data['fragmentation_count'],
            total_frames=self.total_frames,
            active_tracks=avg_active
        )

        return {
            "summary": {
                "identity_stability_score": stability_score,
                "visible_id_switches": switches_data['visible_id_switches'],
                "box_continuity_switches": box_cont_data['box_continuity_switches'],
                "display_id_changes": box_cont_data['display_id_changes'],
                "total_visible_switches": total_visible_switches,
                "switch_breakdown": {
                    "replace": switches_data.get('replace_switches', 0),
                    "swap": switches_data.get('swap_switches', 0),
                    "flicker": switches_data.get('flicker_switches', 0),
                    "box_continuity": box_cont_data['box_continuity_switches'],
                    "display_id_change": box_cont_data['display_id_changes'],
                },
                "duplicate_box_frames": dupes_data['duplicate_frame_count'],
                "teleportation_events": teleport_data['teleportation_count'],
                "fragmentation_count": frag_data['fragmentation_count']
            },
            "detailed": {
                "visible_switches": switches_data,
                "box_continuity": box_cont_data,
                "duplicates": dupes_data,
                "teleportation": teleport_data,
                "fragmentation": frag_data
            }
        }

    def get_forensic_report(self) -> str:
        """Generate a human-readable forensic report of all detected switches."""
        return self.visible_id_switches.get_forensic_report()
