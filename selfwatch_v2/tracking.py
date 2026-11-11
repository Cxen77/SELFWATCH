"""Motion/IoU local baseline. IDs are scoped by camera and stream epoch.

No cross-camera identity claim; unobserved predictions are not rendered.
"""
from dataclasses import dataclass
import math


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


@dataclass
class Track:
    local_id: int
    box: tuple
    last_seen: float
    velocity: tuple = (0., 0.)
    hits: int = 1
    history: tuple = ()

    def predicted_box(self, timestamp):
        dt = timestamp - self.last_seen
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        return (self.box[0]+dx, self.box[1]+dy, self.box[2]+dx, self.box[3]+dy)


class LocalTracker:
    def __init__(self, max_gap_s=2.0, min_iou=.05):
        self.max_gap_s, self.min_iou = max_gap_s, min_iou
        self.scope, self.previous_time = None, None
        self.next_id, self.tracks = 1, {}

    def update(self, frame, instances):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        scope = (frame.camera_id, frame.stream_epoch)
        if self.scope is not None and scope[0] != self.scope[0]:
            raise ValueError("Use a separate local tracker for each camera")
        if scope != self.scope:
            self.scope, self.previous_time, self.next_id, self.tracks = scope, None, 1, {}
        now = frame.captured_at_s
        if self.previous_time is not None and now <= self.previous_time:
            raise ValueError("Local tracker requires strictly increasing capture times")
        self.previous_time = now
        self.tracks = {key: track for key, track in self.tracks.items() if now - track.last_seen <= self.max_gap_s}
        people = [(i, instance) for i, instance in enumerate(instances) if instance.label.lower() == "person"]
        existing = list(self.tracks.values())
        assignments = {}
        if people and existing:
            costs = np.full((len(existing), len(people)), 1e6)
            for row, track in enumerate(existing):
                predicted = track.predicted_box(now)
                pc = ((predicted[0]+predicted[2])/2, (predicted[1]+predicted[3])/2)
                scale = max(1, math.hypot(predicted[2]-predicted[0], predicted[3]-predicted[1]))
                for col, (_, person) in enumerate(people):
                    box = person.box_xyxy
                    distance = math.hypot((box[0]+box[2])/2-pc[0], (box[1]+box[3])/2-pc[1])/scale
                    overlap = iou(predicted, box)
                    if distance <= 1.5 and (overlap >= self.min_iou or distance < .4):
                        costs[row, col] = 1-overlap + .25*distance
            rows, cols = linear_sum_assignment(costs)
            for row, col in zip(rows, cols):
                if costs[row, col] < 1e6:
                    assignments[col] = existing[row]
        result = []
        for col, (index, person) in enumerate(people):
            box = person.box_xyxy
            center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            track = assignments.get(col)
            if track is None:
                track = Track(self.next_id, box, now)
                self.next_id += 1
                self.tracks[track.local_id] = track
            else:
                dt = now-track.last_seen
                old_center = ((track.box[0]+track.box[2])/2, (track.box[1]+track.box[3])/2)
                velocity = tuple((a-b)/dt for a, b in zip(center, old_center))
                track.velocity = tuple(.5*a+.5*b for a, b in zip(track.velocity, velocity))
                track.box, track.last_seen, track.hits = box, now, track.hits+1
            track.history = (*track.history[-63:], (now, *center))
            result.append(dict(instance_index=index, camera_id=scope[0], stream_epoch=scope[1],
                               local_id=track.local_id, hits=track.hits, trajectory=track.history,
                               status="observed", coordinate_space="image_pixels"))
        return result
