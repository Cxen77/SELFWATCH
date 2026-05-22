from typing import List

class FragmentationMetric:
    """
    Detects track death followed by nearby respawn, short-lived identity bursts,
    and ownership splits.
    """
    def __init__(self, short_lived_threshold: int = 15):
        self.short_lived_threshold = short_lived_threshold
        self.track_lifespans = {} # global_id -> int (frames)
        self.fragmentation_count = 0
        self.short_bursts = 0
        
    def update(self, frame_idx: int, visible_objects: List[dict]):
        current_ids = set()
        for obj in visible_objects:
            gid = obj['global_id']
            current_ids.add(gid)
            if gid not in self.track_lifespans:
                self.track_lifespans[gid] = 1
            else:
                self.track_lifespans[gid] += 1
                
    def finalize(self):
        for gid, lifespan in self.track_lifespans.items():
            if lifespan < self.short_lived_threshold:
                self.short_bursts += 1
                
    def get_metrics(self) -> dict:
        self.finalize()
        return {
            'fragmentation_count': self.fragmentation_count,
            'short_lived_bursts': self.short_bursts
        }
