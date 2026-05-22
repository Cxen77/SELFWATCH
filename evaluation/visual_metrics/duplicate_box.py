from typing import List, Dict

class DuplicateBoxMetric:
    """
    Detects when 2 or more bounding boxes represent the same person visually.
    Measures frame count, duration, and severity.
    """
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.duplicate_frame_count = 0
        self.total_duplicate_events = 0
        self.duplicate_events = []
        
    def compute_iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
            
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, frame_idx: int, visible_objects: List[dict]):
        """
        visible_objects: list of dicts with 'global_id', 'state' (ACTIVE, THINKING), 'bbox'
        """
        has_duplicate_this_frame = False
        n = len(visible_objects)
        
        for i in range(n):
            for j in range(i + 1, n):
                obj1 = visible_objects[i]
                obj2 = visible_objects[j]
                
                iou = self.compute_iou(obj1['bbox'], obj2['bbox'])
                if iou > self.iou_threshold:
                    has_duplicate_this_frame = True
                    self.total_duplicate_events += 1
                    self.duplicate_events.append({
                        'frame': frame_idx,
                        'id1': obj1['global_id'],
                        'state1': obj1.get('state', 'UNKNOWN'),
                        'id2': obj2['global_id'],
                        'state2': obj2.get('state', 'UNKNOWN'),
                        'iou': iou
                    })
                    
        if has_duplicate_this_frame:
            self.duplicate_frame_count += 1

    def get_metrics(self) -> dict:
        return {
            'duplicate_frame_count': self.duplicate_frame_count,
            'total_duplicate_events': self.total_duplicate_events,
            'duplicate_events': self.duplicate_events
        }
