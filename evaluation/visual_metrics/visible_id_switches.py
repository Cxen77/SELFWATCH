import numpy as np
from typing import Dict, List, Tuple

class VisibleIDSwitchMetric:
    """
    Measures ID switches that are actually visible on-screen, ignoring internal
    hidden rebindings that are repaired before visualization.
    """
    def __init__(self, spatial_tolerance: float = 50.0):
        self.spatial_tolerance = spatial_tolerance
        self.visible_switch_count = 0
        self.last_frame_positions = {}  # {global_id: (cx, cy)}
        self.switch_events = []
        
    def update(self, frame_idx: int, visible_objects: List[dict]):
        """
        visible_objects: list of dicts with keys: 'global_id', 'bbox' (x1,y1,x2,y2)
        """
        current_positions = {}
        
        for obj in visible_objects:
            gid = obj['global_id']
            bbox = obj['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            current_positions[gid] = (cx, cy)
            
        # Detect visible ID replacement
        # If a global_id disappears and a new one appears very close by in the next frame
        disappeared = set(self.last_frame_positions.keys()) - set(current_positions.keys())
        appeared = set(current_positions.keys()) - set(self.last_frame_positions.keys())
        
        for d_id in disappeared:
            d_pos = self.last_frame_positions[d_id]
            for a_id in appeared:
                a_pos = current_positions[a_id]
                dist = np.hypot(d_pos[0] - a_pos[0], d_pos[1] - a_pos[1])
                if dist < self.spatial_tolerance:
                    self.visible_switch_count += 1
                    self.switch_events.append({
                        'frame': frame_idx,
                        'old_id': d_id,
                        'new_id': a_id,
                        'distance': dist
                    })
                    break # Assuming one-to-one replacement
                    
        self.last_frame_positions = current_positions

    def get_metrics(self) -> dict:
        return {
            'visible_id_switches': self.visible_switch_count,
            'switch_events': self.switch_events
        }
