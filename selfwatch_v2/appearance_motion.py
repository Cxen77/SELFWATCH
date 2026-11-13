"""Camera-local motion and appearance association for fixed-input replay."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracking import iou


@dataclass(frozen=True)
class LocalPolicy:
    cosine: float = .64
    birth_score: float = .28
    appearance_weight: float = 1.4
    max_gap_s: float = 1.5


@dataclass
class Tracklet:
    box: np.ndarray
    time_s: float
    velocity: np.ndarray
    appearance: np.ndarray | None
    hits: int = 1


class AppearanceMotion:
    """Assign each measured detection to at most one observed owner."""
    def __init__(self, policy):
        self.policy = policy
        self.tracks = {}
        self.next_id = 0
        self.epoch = None
        self.last = -math.inf

    def update(self, frame, instances, vectors):
        if self.epoch != frame.stream_epoch:
            self.epoch = frame.stream_epoch
            self.tracks.clear()
            self.last = -math.inf
        now = frame.captured_at_s
        if now <= self.last:
            raise ValueError("Expected increasing camera capture times")
        self.last = now
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track.time_s <= self.policy.max_gap_s}
        people = [(i, np.array(obs.box_xyxy), obs.score) for i, obs in enumerate(instances)
                  if obs.label == "person"]
        owners = list(self.tracks)
        costs = np.full((len(owners), len(people)), 1e6)
        for row, key in enumerate(owners):
            track = self.tracks[key]
            age = now - track.time_s
            prediction = track.box + track.velocity * min(age, .65)
            if np.any(prediction[2:] <= prediction[:2]):
                prediction = track.box
            scale = max(16, np.linalg.norm(track.box[2:] - track.box[:2]))
            for col, (index, box, _) in enumerate(people):
                distance = np.linalg.norm((prediction[:2] + prediction[2:] - box[:2] - box[2:]) / 2) / scale
                overlap = iou(prediction, box)
                feature = vectors.get(index)
                cosine = float(track.appearance @ feature) if track.appearance is not None and feature is not None else None
                appearance = cosine is not None and cosine >= self.policy.cosine
                # A distant or stale prediction needs appearance support.
                # Very close measured boxes can sustain a track without a crop.
                if distance > 1.25 or (cosine is not None and cosine < .33):
                    continue
                if not appearance and (age > .65 or (overlap < .15 and distance > .28)):
                    continue
                cost = .7 * (1 - overlap) + .40 * distance + .10 * age
                cost += self.policy.appearance_weight * (1 - cosine if cosine is not None else .42)
                costs[row, col] = cost
        rows, cols = linear_sum_assignment(costs)
        assignments = {col: owners[row] for row, col in zip(rows, cols) if costs[row, col] < 1e6}
        result = []
        for col, (index, box, score) in enumerate(people):
            key = assignments.get(col)
            feature = vectors.get(index)
            if key is None:
                if score < self.policy.birth_score:
                    continue
                self.next_id += 1
                key = self.next_id
                self.tracks[key] = Tracklet(box, now, np.zeros(4), feature)
            else:
                track = self.tracks[key]
                age = now - track.time_s
                track.velocity = .65 * (box - track.box) / age + .35 * track.velocity
                track.box, track.time_s = box, now
                track.hits += 1
                if feature is not None:
                    if track.appearance is None:
                        track.appearance = feature
                    elif track.appearance @ feature >= self.policy.cosine:
                        mean = .85 * track.appearance + .15 * feature
                        track.appearance = mean / np.linalg.norm(mean)
            result.append(dict(local_id=key, instance_index=index, hits=self.tracks[key].hits, status="observed"))
        return result
