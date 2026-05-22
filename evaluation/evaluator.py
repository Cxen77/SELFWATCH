from evaluation.visual_metrics.visible_id_switches import VisibleIDSwitchMetric
from evaluation.visual_metrics.duplicate_box import DuplicateBoxMetric
from evaluation.visual_metrics.fragmentation import FragmentationMetric
from evaluation.teleportation_metrics.teleportation import IdentityTeleportationMetric
from evaluation.stability_metrics.occlusion_recovery import OcclusionRecoveryMetric
from evaluation.stability_metrics.identity_stability import IdentityStabilityScore

class SELFWATCHEvaluator:
    """
    Master evaluator class that orchestrates all visible continuity and stability metrics.
    """
    def __init__(self):
        self.visible_id_switches = VisibleIDSwitchMetric()
        self.duplicate_box = DuplicateBoxMetric()
        self.fragmentation = FragmentationMetric()
        self.teleportation = IdentityTeleportationMetric()
        self.occlusion_recovery = OcclusionRecoveryMetric()
        self.stability_scorer = IdentityStabilityScore()
        
        self.total_frames = 0
        self.active_tracks_accumulated = 0
        
    def update(self, frame_idx: int, visible_rendered_identities: list, 
               tracks: list = None, detections: list = None, 
               suppression_regions: list = None, frozen_gids: list = None):
        """
        Process a single frame's visual output and tracking state.
        """
        self.total_frames += 1
        self.active_tracks_accumulated += len(visible_rendered_identities)
        
        self.visible_id_switches.update(frame_idx, visible_rendered_identities)
        self.duplicate_box.update(frame_idx, visible_rendered_identities)
        self.fragmentation.update(frame_idx, visible_rendered_identities)
        self.teleportation.update(frame_idx, visible_rendered_identities)
        
    def get_final_report(self):
        switches_data = self.visible_id_switches.get_metrics()
        dupes_data = self.duplicate_box.get_metrics()
        teleport_data = self.teleportation.get_metrics()
        frag_data = self.fragmentation.get_metrics()
        
        avg_active = self.active_tracks_accumulated / max(1, self.total_frames)
        
        stability_score = self.stability_scorer.compute_score(
            visible_switches=switches_data['visible_id_switches'],
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
                "duplicate_box_frames": dupes_data['duplicate_frame_count'],
                "teleportation_events": teleport_data['teleportation_count'],
                "fragmentation_count": frag_data['fragmentation_count']
            },
            "detailed": {
                "visible_switches": switches_data,
                "duplicates": dupes_data,
                "teleportation": teleport_data,
                "fragmentation": frag_data
            }
        }
