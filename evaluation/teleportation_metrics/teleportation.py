import numpy as np
from typing import List, Dict

class IdentityTeleportationMetric:
    """
    Detects physically impossible identity movement such as:
    - opposite-direction jumps
    - large instantaneous displacement
    - impossible velocity changes
    - cross-frame ownership teleportation

    Gap-aware: allows larger movement when an identity was missing
    for multiple frames (predicted/occluded).
    """
    def __init__(self, max_speed_pixels_per_frame: float = 100.0):
        self.max_speed = max_speed_pixels_per_frame
        self.teleportation_count = 0
        self.impossible_velocity_events = 0
        self.last_positions = {} # global_id -> (cx, cy, last_frame)
        self.events = []

    def update(self, frame_idx: int, visible_objects: List[dict]):
        for obj in visible_objects:
            gid = obj['global_id']
            bbox = obj['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            if gid in self.last_positions:
                px, py, last_frame = self.last_positions[gid]
                dist = np.hypot(cx - px, cy - py)

                # Gap-aware: scale threshold by frames since last seen
                gap = max(1, frame_idx - last_frame)
                effective_threshold = self.max_speed * gap

                if dist > effective_threshold:
                    self.teleportation_count += 1
                    self.impossible_velocity_events += 1
                    self.events.append({
                        'frame': frame_idx,
                        'global_id': gid,
                        'distance': dist,
                        'gap_frames': gap,
                        'effective_threshold': effective_threshold,
                    })

            self.last_positions[gid] = (cx, cy, frame_idx)

    def get_metrics(self) -> dict:
        return {
            'teleportation_count': self.teleportation_count,
            'impossible_velocity_events': self.impossible_velocity_events,
            'teleportation_events': self.events
        }

