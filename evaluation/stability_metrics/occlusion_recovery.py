from typing import List

class OcclusionRecoveryMetric:
    """
    Measures successful identity preservation through occlusion,
    wrong post-occlusion recovery, and recovery latency.
    """
    def __init__(self):
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.recovery_latency_sum = 0
        self.occluded_tracks = {} # global_id -> frame_occluded

    def track_occluded(self, global_id: int, frame_idx: int):
        self.occluded_tracks[global_id] = frame_idx

    def track_recovered(self, global_id: int, frame_idx: int, correct: bool = True):
        if global_id in self.occluded_tracks:
            latency = frame_idx - self.occluded_tracks[global_id]
            self.recovery_latency_sum += latency
            if correct:
                self.successful_recoveries += 1
            else:
                self.failed_recoveries += 1
            del self.occluded_tracks[global_id]

    def get_metrics(self) -> dict:
        total_recoveries = self.successful_recoveries + self.failed_recoveries
        avg_latency = self.recovery_latency_sum / total_recoveries if total_recoveries > 0 else 0
        
        return {
            'successful_recoveries': self.successful_recoveries,
            'failed_recoveries': self.failed_recoveries,
            'average_recovery_latency_frames': avg_latency
        }
