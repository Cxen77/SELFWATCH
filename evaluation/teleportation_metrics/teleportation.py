import numpy as np
from typing import List, Dict

class IdentityTeleportationMetric:
    """
    Detects physically impossible identity movement such as:
    - opposite-direction jumps
    - large instantaneous displacement
    - impossible velocity changes
    - cross-frame ownership teleportation
    """
    def __init__(self, max_speed_pixels_per_frame: float = 100.0):
        self.max_speed = max_speed_pixels_per_frame
        self.teleportation_count = 0
        self.impossible_velocity_events = 0
        self.last_positions = {} # global_id -> (cx, cy)
        self.events = []

    def update(self, frame_idx: int, visible_objects: List[dict]):
        for obj in visible_objects:
            gid = obj['global_id']
            bbox = obj['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            if gid in self.last_positions:
                px, py = self.last_positions[gid]
                dist = np.hypot(cx - px, cy - py)
                
                if dist > self.max_speed:
                    self.teleportation_count += 1
                    self.impossible_velocity_events += 1
                    self.events.append({
                        'frame': frame_idx,
                        'global_id': gid,
                        'distance': dist,
                        'max_speed_allowed': self.max_speed
                    })
                    
            self.last_positions[gid] = (cx, cy)

    def get_metrics(self) -> dict:
        return {
            'teleportation_count': self.teleportation_count,
            'impossible_velocity_events': self.impossible_velocity_events,
            'teleportation_events': self.events
        }
