"""
SELFWATCH — Phantom Tracking System (v2)

Maintains invisible spatial predictions for lost tracks using
trajectory-cone prediction instead of circular search.

Key improvements over v1:
  - Forward motion cone replaces isotropic circular search radius
  - Hard direction-consistency gating: dot(v_old, v_new) < 0 → reject
  - Phantom confidence decays faster when no directional agreement
  - Ownership lock: thin/static occluders keep phantom alive longer
  - Velocity stored from OBSERVED positions, not Kalman predictions

Cost: One vector addition per phantom per frame. Negligible.
"""

import time
import math
import numpy as np


# ── Cone parameters ──────────────────────────────────────────────────
CONE_HALF_ANGLE_DEG = 45.0     # half-angle of the forward search cone
CONE_HALF_ANGLE_RAD = math.radians(CONE_HALF_ANGLE_DEG)
CONE_COS_THRESHOLD = math.cos(CONE_HALF_ANGLE_RAD)  # ≈ 0.707 for 45°

# If speed is below this, we fall back to circular search (stationary target)
STATIONARY_SPEED_THRESH = 0.5  # px/frame


class PhantomTrack:
    """
    Invisible spatial prediction for a lost identity.

    Attributes:
        track_id:              Original global track ID.
        embedding:             Last known L2-normalized embedding.
        gallery:               List of diverse embeddings from active identity.
        position:              Predicted [x1, y1, x2, y2] (updated each frame).
        velocity:              EWMA-smoothed [vx, vy] pixels per frame.
        initial_velocity:      Velocity at spawn time (immutable, for cone).
        initial_speed:         Speed at spawn time.
        confidence:            Decays over time (uncertainty grows).
        search_radius:         Fallback circular radius for stationary phantoms.
        birth_time:            When the phantom was spawned.
        importance:            How valuable this identity is (based on lifetime).
        age_frames:            Frames since phantom was created.
        last_motion_agreement: Dot product from most recent match attempt.
        is_static_occluder:    True if track was lost near a thin/static edge.
    """

    def __init__(self, track_id, embedding, last_position, velocity,
                 importance=1.0, gallery=None, is_static_occluder=False):
        self.track_id = track_id
        self.embedding = embedding.copy()
        self.gallery = [g.copy() for g in gallery] if gallery else []
        self.position = np.array(last_position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.initial_velocity = self.velocity.copy()
        self.initial_speed = float(np.linalg.norm(self.initial_velocity))
        self.confidence = 1.0
        self.search_radius = 80.0       # fallback for stationary phantoms
        self.birth_time = time.perf_counter()
        self.importance = importance
        self.age_frames = 0
        self.last_motion_agreement = 0.0
        self.is_static_occluder = is_static_occluder

        # Pre-compute normalized direction for cone checks
        if self.initial_speed > STATIONARY_SPEED_THRESH:
            self._direction = self.initial_velocity / self.initial_speed
        else:
            self._direction = None  # stationary → use circular search

    def predict(self, frame_delta=1):
        """Advance phantom using frame-gap-aware velocity model."""
        frame_delta = max(1, int(frame_delta))
        self.age_frames += frame_delta

        # Move center by velocity
        cx = (self.position[0] + self.position[2]) / 2.0 + self.velocity[0] * frame_delta
        cy = (self.position[1] + self.position[3]) / 2.0 + self.velocity[1] * frame_delta
        w = self.position[2] - self.position[0]
        h = self.position[3] - self.position[1]

        self.position = np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dtype=np.float32)

        # ── Confidence decay ─────────────────────────────────────────
        # Base decay
        base_decay = 0.96 ** frame_delta

        # Extra penalty if no recent motion agreement
        if self.last_motion_agreement <= 0:
            base_decay *= 0.92 ** frame_delta

        # Static occluder bonus: slower decay for thin poles/shelves
        if self.is_static_occluder:
            base_decay = max(base_decay, 0.98 ** frame_delta)

        self.confidence *= base_decay

        # Search radius grows, but slower for moving phantoms (cone handles it)
        if self._direction is not None:
            self.search_radius = 60.0 + self.age_frames * 4.0
        else:
            self.search_radius = 80.0 + self.age_frames * 8.0

        # Dampen velocity over time (friction)
        self.velocity *= 0.95 ** frame_delta

    @property
    def center(self):
        return np.array([
            (self.position[0] + self.position[2]) / 2.0,
            (self.position[1] + self.position[3]) / 2.0,
        ])

    @property
    def is_moving(self):
        """True if the phantom was spawned with meaningful velocity."""
        return self._direction is not None

    def point_in_cone(self, px, py):
        """
        Check if point (px, py) lies within the forward motion cone
        of this phantom.

        The cone is centered at the phantom's predicted center,
        opening in the direction of initial_velocity, with half-angle
        CONE_HALF_ANGLE_DEG.

        Returns:
            (in_cone: bool, cos_angle: float)
        """
        if self._direction is None:
            # Stationary phantom: use circular search
            pcx, pcy = self.center
            dist = math.hypot(px - pcx, py - pcy)
            return dist <= self.search_radius, 0.0

        pcx, pcy = self.center
        dx = px - pcx
        dy = py - pcy
        dist = math.hypot(dx, dy)

        if dist < 1e-3:
            return True, 1.0   # on top of predicted center

        # Unit vector from phantom center to candidate point
        to_point = np.array([dx / dist, dy / dist], dtype=np.float32)
        cos_angle = float(np.dot(self._direction, to_point))

        # Must be within the cone half-angle AND within search radius
        in_cone = cos_angle >= CONE_COS_THRESHOLD and dist <= self.search_radius
        return in_cone, cos_angle

    def get_cone_tip_and_edges(self, length=None):
        """
        Return (tip, left_edge, right_edge) for forensic overlay drawing.
        All coordinates are absolute pixel positions.
        """
        if self._direction is None:
            return None

        tip = self.center
        if length is None:
            length = self.search_radius

        # Rotate direction by ±CONE_HALF_ANGLE_RAD
        cos_a = math.cos(CONE_HALF_ANGLE_RAD)
        sin_a = math.sin(CONE_HALF_ANGLE_RAD)
        dx, dy = float(self._direction[0]), float(self._direction[1])

        left = tip + np.array([
            (dx * cos_a - dy * sin_a) * length,
            (dx * sin_a + dy * cos_a) * length,
        ])
        right = tip + np.array([
            (dx * cos_a + dy * sin_a) * length,
            (-dx * sin_a + dy * cos_a) * length,
        ])

        return tip, left, right

    def is_expired(self, max_age=90):
        """Phantom dies after max_age frames (~3s at 30fps)."""
        effective_max_age = max_age
        if self.is_static_occluder:
            effective_max_age = int(max_age * 1.5)  # 50% longer for poles/shelves
        return self.age_frames >= effective_max_age or self.confidence < 0.10


class PhantomTracker:
    """
    Manages all active phantoms and provides instant-match retrieval
    using trajectory-cone gating and direction-consistency checks.

    Usage:
        phantom_tracker = PhantomTracker(max_phantom_age=90)

        # When a track is lost:
        phantom_tracker.spawn(track_id, embedding, position, velocity, ...)

        # Every frame:
        phantom_tracker.tick()   # advance all phantoms

        # When a new detection appears:
        match = phantom_tracker.try_match(new_embedding, new_position, new_velocity)
        if match:
            recovered_id = match.track_id
    """

    def __init__(self, max_phantom_age=90, match_threshold=0.80,
                 event_logger=None, metrics=None):
        self.phantoms = {}              # track_id -> PhantomTrack
        self.max_phantom_age = max_phantom_age
        self.match_threshold = match_threshold
        self._logger = event_logger
        self._metrics = metrics

    def spawn(self, track_id, embedding, last_position, velocity,
              importance=1.0, gallery=None, is_static_occluder=False):
        """Create a phantom for a lost track."""
        vel = np.array(velocity, dtype=np.float32) if velocity is not None else np.zeros(2)
        speed = np.linalg.norm(vel)
        if speed < 0.1:
            vel = np.zeros(2, dtype=np.float32)

        self.phantoms[track_id] = PhantomTrack(
            track_id=track_id,
            embedding=embedding,
            last_position=last_position,
            velocity=vel,
            importance=importance,
            gallery=gallery,
            is_static_occluder=is_static_occluder,
        )

        if self._logger:
            self._logger.log("phantom_spawn", track_id,
                             speed=round(float(speed), 2),
                             static_occ=is_static_occluder)

    def tick(self, frame_delta=1):
        """Advance all phantoms one frame and expire dead ones."""
        frame_delta = max(1, int(frame_delta))
        expired = []
        for tid, phantom in self.phantoms.items():
            phantom.predict(frame_delta=frame_delta)
            if phantom.is_expired(self.max_phantom_age):
                expired.append(tid)

        for tid in expired:
            if self._logger:
                self._logger.log("phantom_expired", tid,
                                 age=self.phantoms[tid].age_frames,
                                 conf=round(self.phantoms[tid].confidence, 3))
            del self.phantoms[tid]

    def try_match(self, new_embedding, new_position, new_velocity=None):
        """
        Try to match a new detection against active phantoms.

        Uses trajectory-cone prediction (for moving phantoms) or
        circular proximity (for stationary phantoms), combined with
        embedding similarity and hard direction-consistency gating.

        Direction gate: if dot(phantom_velocity, new_velocity) < 0,
        the match is REJECTED — prevents identity teleportation.

        Args:
            new_embedding:  L2-normalized 512-dim vector.
            new_position:   [x1, y1, x2, y2] bounding box.
            new_velocity:   [vx, vy] velocity vector or None.

        Returns:
            PhantomTrack or None.
        """
        if not self.phantoms:
            return None

        new_cx = (new_position[0] + new_position[2]) / 2.0
        new_cy = (new_position[1] + new_position[3]) / 2.0

        best_phantom = None
        best_score = 0.0

        emb = new_embedding.copy()
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm

        for tid, phantom in self.phantoms.items():

            # ── Spatial gate: trajectory cone or circular ─────────────
            if phantom.is_moving:
                in_cone, cone_cos = phantom.point_in_cone(new_cx, new_cy)
                if not in_cone:
                    continue
            else:
                pcx, pcy = phantom.center
                dist = math.hypot(new_cx - pcx, new_cy - pcy)
                if dist > phantom.search_radius:
                    continue
                cone_cos = 0.0  # neutral for stationary

            # ── Embedding similarity ─────────────────────────────────
            sims = [float(np.dot(emb, phantom.embedding))]
            for g in phantom.gallery:
                sims.append(float(np.dot(emb, g)))

            best_sim = max(sims)

            if best_sim < self.match_threshold:
                continue

            # ── Hard direction-consistency gating ─────────────────────
            dot_prod = 0.0
            motion_score = 0.5  # neutral default

            if new_velocity is not None and phantom.initial_speed > STATIONARY_SPEED_THRESH:
                v_new = np.array(new_velocity, dtype=np.float32)
                speed_new = float(np.linalg.norm(v_new))

                if speed_new > STATIONARY_SPEED_THRESH:
                    v_new_norm = v_new / speed_new
                    dot_prod = float(np.dot(phantom._direction, v_new_norm))

                    # HARD GATE 1: opposite direction → reject immediately
                    if dot_prod < 0:
                        if self._logger:
                            self._logger.log(
                                "phantom_reject_direction", phantom.track_id,
                                dot=round(dot_prod, 3),
                                phantom_vel=phantom.initial_velocity.tolist(),
                                new_vel=v_new.tolist())
                        continue

                    # HARD GATE 2: speed consistency (reject if >3x different)
                    speed_ratio = max(speed_new, phantom.initial_speed) / max(min(speed_new, phantom.initial_speed), 0.1)
                    if speed_ratio > 3.0:
                        if self._logger:
                            self._logger.log(
                                "phantom_reject_speed", phantom.track_id,
                                ratio=round(speed_ratio, 2))
                        continue

                    # HARD GATE 3: require minimum heading agreement for old phantoms
                    if phantom.age_frames > 15 and dot_prod < 0.3:
                        if self._logger:
                            self._logger.log(
                                "phantom_reject_weak_heading", phantom.track_id,
                                dot=round(dot_prod, 3), age=phantom.age_frames)
                        continue

                    # Soft scale: [0, 1] → motion_score in [0.5, 1.0]
                    motion_score = 0.5 + 0.5 * dot_prod
                else:
                    # New detection is nearly stationary — reduce confidence
                    motion_score = 0.35

            # ── Score fusion ─────────────────────────────────────────
            pcx, pcy = phantom.center
            dist = math.hypot(new_cx - pcx, new_cy - pcy)
            proximity = max(0.0, 1.0 - dist / max(phantom.search_radius, 1.0))

            # Cone bonus: if inside cone AND well-aligned, boost score
            cone_bonus = max(0.0, cone_cos) * 0.1 if phantom.is_moving else 0.0

            score = (0.50 * best_sim +
                     0.15 * proximity +
                     0.15 * phantom.confidence +
                     0.10 * motion_score +
                     0.10 * cone_bonus)

            # Apply motion confidence as a multiplier to penalize poor agreement
            score *= (0.5 + 0.5 * motion_score)

            phantom.last_motion_agreement = dot_prod

            if score > best_score:
                best_score = score
                best_phantom = phantom

        if best_phantom is not None:
            if self._logger:
                self._logger.log("phantom_match", best_phantom.track_id,
                                 score=round(best_score, 3),
                                 age=best_phantom.age_frames,
                                 direction_dot=round(best_phantom.last_motion_agreement, 3))
            return best_phantom

        return None

    def remove(self, track_id):
        """Remove a phantom after successful resurrection."""
        self.phantoms.pop(track_id, None)

    @property
    def count(self):
        return len(self.phantoms)

    def get_all_positions(self):
        """Return phantom positions for debug overlay."""
        return {tid: p.position.tolist() for tid, p in self.phantoms.items()}
