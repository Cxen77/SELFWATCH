"""
Occlusion Group Manager — Aggressive Ownership Lock

When two or more tracked people overlap, their identities are FROZEN:
  - No identity proposals accepted (prevents ID swap during crossing)
  - Appearance features NOT updated (prevents embedding pollution)
  - Motion/trajectory prediction continues normally
  - On separation (+SEPARATION_COOLDOWN frames), normal tracking resumes
  - After unfreeze, a POST_FREEZE_COOLDOWN blocks rapid re-proposals
  - Exit trajectory predictions are cached for plausible recovery

Static occlusion detection:
  - Tracks that disappear while near a confirmed track's edge are
    flagged as potentially occluded by a thin/static object (pole, shelf)
  - These get extended ownership lock (STATIC_COOLDOWN frames)

This is the single highest-impact fix for ID switching during
path crossing and crowd overlap scenarios.

Cost: O(N^2) pairwise IoU check where N = confirmed tracks.
For typical retail scenes (N < 20), this is < 0.01ms per frame.
"""

import numpy as np
import math


class OcclusionGroupManager:
    OVERLAP_IOU_THRESH = 0.08      # IoU threshold to trigger freeze (lowered for earlier detection)
    SEPARATION_COOLDOWN = 12       # Frames after separation before unfreeze (was 8)
    STATIC_COOLDOWN = 20           # Extended cooldown for static occluders
    POST_FREEZE_COOLDOWN = 8       # Frames after unfreeze where rebinding is still restricted
    FROZEN_REBIND_MIN_FRAMES = 5   # Minimum consistent frames before reassignment allowed post-freeze

    def __init__(self):
        self._overlap_counter = {}  # gid -> frames_since_last_overlap
        self._overlap_type = {}     # gid -> "crossing" or "static"

        # ── Frozen cooldown tracking ────────────────────────────────────
        # gid -> frames remaining in post-freeze cooldown
        self._post_freeze_cooldown = {}

        # ── Exit trajectory cache ───────────────────────────────────────
        # gid -> { velocity, center, box, freeze_frame }
        # Cached when a gid enters frozen state, used for plausible exit recovery
        self._exit_trajectories = {}

        self._frame_count = 0

    def update(self, confirmed_boxes, velocities=None):
        """
        Detect overlapping tracks and return frozen gid set.

        Args:
            confirmed_boxes: dict of gid -> [x1, y1, x2, y2]
            velocities: optional dict of gid -> [vx, vy] (for exit trajectory caching)

        Returns:
            set of gids that should be FROZEN (no ID changes, no emb updates)
        """
        self._frame_count += 1
        velocities = velocities or {}

        currently_overlapping = set()
        overlap_pairs = []

        gids = list(confirmed_boxes.keys())
        if len(gids) < 2:
            # Age out any existing counters
            self._age_out(currently_overlapping)
            self._tick_post_freeze_cooldowns()
            return self.frozen_gids

        boxes = [confirmed_boxes[g] for g in gids]

        # Pairwise IoU check
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                iou = self._iou(boxes[i], boxes[j])
                if iou > self.OVERLAP_IOU_THRESH:
                    currently_overlapping.add(gids[i])
                    currently_overlapping.add(gids[j])
                    overlap_pairs.append((gids[i], gids[j], iou))

        # Classify overlap type
        for gid_a, gid_b, iou in overlap_pairs:
            # High IoU = path crossing (people walking through each other)
            # Low IoU = edge occlusion (person behind pole/shelf)
            if iou < 0.25:
                # Partial overlap → likely static occluder
                self._overlap_type[gid_a] = "static"
                self._overlap_type[gid_b] = "static"
            else:
                self._overlap_type.setdefault(gid_a, "crossing")
                self._overlap_type.setdefault(gid_b, "crossing")

        # Cache exit trajectory for newly frozen gids
        for gid in currently_overlapping:
            if gid not in self._overlap_counter:
                # Newly entering frozen state — cache trajectory for exit prediction
                box = confirmed_boxes.get(gid)
                vel = velocities.get(gid)
                if box is not None:
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    self._exit_trajectories[gid] = {
                        "velocity": list(vel) if vel is not None else [0.0, 0.0],
                        "center": [cx, cy],
                        "box": list(box),
                        "freeze_frame": self._frame_count,
                    }

        # Reset counter for currently overlapping
        for gid in currently_overlapping:
            self._overlap_counter[gid] = 0

        self._age_out(currently_overlapping)
        self._tick_post_freeze_cooldowns()
        return self.frozen_gids

    def _age_out(self, currently_overlapping):
        """Age non-overlapping gids and remove expired ones."""
        expired = []
        for gid in self._overlap_counter:
            if gid not in currently_overlapping:
                self._overlap_counter[gid] += 1
                # Use longer cooldown for static occluder type
                cooldown = self.STATIC_COOLDOWN if self._overlap_type.get(gid) == "static" else self.SEPARATION_COOLDOWN
                if self._overlap_counter[gid] > cooldown:
                    expired.append(gid)
        for gid in expired:
            del self._overlap_counter[gid]
            self._overlap_type.pop(gid, None)
            # Start post-freeze cooldown when fully unfreezing
            self._post_freeze_cooldown[gid] = self.POST_FREEZE_COOLDOWN

    def _tick_post_freeze_cooldowns(self):
        """Decrement post-freeze cooldowns and remove expired ones."""
        expired = []
        for gid in self._post_freeze_cooldown:
            self._post_freeze_cooldown[gid] -= 1
            if self._post_freeze_cooldown[gid] <= 0:
                expired.append(gid)
        for gid in expired:
            del self._post_freeze_cooldown[gid]
            # Clean up exit trajectory cache when cooldown fully expires
            self._exit_trajectories.pop(gid, None)

    @property
    def frozen_gids(self):
        """Set of all currently frozen gids (overlapping + cooldown)."""
        return set(self._overlap_counter.keys())

    @property
    def cooldown_gids(self):
        """Set of gids in post-freeze cooldown (recently unfrozen, rebinding restricted)."""
        return set(self._post_freeze_cooldown.keys())

    @property
    def restricted_gids(self):
        """Set of all gids where rebinding is restricted (frozen + cooldown)."""
        return self.frozen_gids | self.cooldown_gids

    def is_frozen(self, gid):
        return gid in self._overlap_counter

    def is_in_cooldown(self, gid):
        """Check if a gid is in post-freeze cooldown (recently unfrozen)."""
        return gid in self._post_freeze_cooldown

    def is_restricted(self, gid):
        """Check if a gid is either frozen or in post-freeze cooldown."""
        return self.is_frozen(gid) or self.is_in_cooldown(gid)

    def get_exit_trajectory(self, gid):
        """
        Get the cached exit trajectory prediction for a frozen/recently-unfrozen gid.

        Returns:
            dict with 'velocity', 'center', 'box', 'freeze_frame' or None
        """
        return self._exit_trajectories.get(gid)

    def predict_exit_position(self, gid, current_frame=None):
        """
        Predict where a frozen identity should emerge based on cached trajectory.

        Returns:
            (pred_cx, pred_cy, max_radius) or None
        """
        traj = self._exit_trajectories.get(gid)
        if traj is None:
            return None

        if current_frame is None:
            current_frame = self._frame_count

        frames_elapsed = max(1, current_frame - traj["freeze_frame"])
        vx, vy = traj["velocity"]
        speed = math.hypot(vx, vy)

        pred_cx = traj["center"][0] + vx * frames_elapsed
        pred_cy = traj["center"][1] + vy * frames_elapsed

        # Radius grows with time and speed uncertainty
        max_radius = min(250.0, 60.0 + frames_elapsed * max(3.0, speed * 1.5))

        return pred_cx, pred_cy, max_radius

    def is_near_exit_region(self, gid, track_cx, track_cy, current_frame=None):
        """
        Check if a track position is near the predicted exit region of a frozen identity.

        Returns:
            (is_near: bool, distance: float)
        """
        pred = self.predict_exit_position(gid, current_frame)
        if pred is None:
            return True, 0.0  # No prediction available, allow

        pred_cx, pred_cy, max_radius = pred
        dist = math.hypot(track_cx - pred_cx, track_cy - pred_cy)
        return dist <= max_radius, dist

    def get_occlusion_type(self, gid):
        """Return 'crossing', 'static', or None."""
        return self._overlap_type.get(gid)

    @property
    def count(self):
        return len(self._overlap_counter)

    @staticmethod
    def _iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / (area_a + area_b - inter + 1e-6)
