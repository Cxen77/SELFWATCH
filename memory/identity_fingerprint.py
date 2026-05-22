"""
Identity Fingerprint — Multi-Signal Person Representation

Builds a rich, temporal identity signature combining:
  1. EWMA embedding (slow-decaying temporal average of OSNet embeddings)
  2. Embedding gallery (diverse historical viewpoints)
  3. Color histogram (upper/lower body HSV distribution)
  4. Body features (aspect ratio, height/width, size)
  5. Motion profile (average speed, direction, velocity variance)

All features are computed from data already available (crops, boxes,
embeddings) with ZERO extra GPU work. Only lightweight CPU operations.

Usage:
    fp = IdentityFingerprint()
    fp.update(embedding, box, crop, score)   # each frame
    score = fp.compare(other_fp_or_embedding) # identity matching
"""

import numpy as np
import cv2
import math


def _l2_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


class IdentityFingerprint:
    """
    Multi-signal identity representation for a single global identity.

    Scoring weights:
      appearance:   0.45  (EWMA embedding vs query)
      historical:   0.20  (gallery max match)
      color:        0.15  (upper+lower body color histogram)
      motion:       0.10  (velocity/direction consistency)
      body:         0.10  (aspect ratio + size similarity)
    """
    # Scoring weights
    W_APPEARANCE = 0.45
    W_HISTORICAL = 0.20
    W_COLOR = 0.15
    W_MOTION = 0.10
    W_BODY = 0.10

    # EWMA decay rates
    EWMA_ALPHA_SLOW = 0.08   # Slow-decaying stable identity signature
    EWMA_ALPHA_FAST = 0.5    # Fast-adapting recent appearance

    # Gallery
    MAX_GALLERY = 6
    GALLERY_DIVERSITY = 0.93  # Min cosine dist to be "diverse"

    # Color histogram bins
    H_BINS = 16
    S_BINS = 8

    def __init__(self):
        # Embedding signals
        self.ewma_embedding = None       # Slow EWMA (identity signature)
        self.recent_embedding = None     # Fast EWMA (recent look)
        self.best_embedding = None       # Highest-confidence embedding
        self.best_score = 0.0
        self.gallery = []                # Diverse embedding gallery

        # Color histograms (upper and lower body, HSV)
        self.upper_color_hist = None     # np.array, normalized
        self.lower_color_hist = None

        # Body features
        self.avg_aspect_ratio = 0.0      # w/h ratio, EWMA
        self.avg_height = 0.0            # pixel height, EWMA
        self.avg_width = 0.0             # pixel width, EWMA

        # Motion profile
        self.avg_speed = 0.0             # pixels/frame, EWMA
        self.avg_direction = 0.0         # radians, circular EWMA
        self.speed_variance = 0.0        # motion consistency
        self.velocity_history = []       # last N velocities

        # Metadata
        self.update_count = 0

    # ── UPDATE ────────────────────────────────────────────────────────

    def update(self, embedding, box, crop=None, score=0.5, velocity=None):
        """
        Update fingerprint with new observation.

        Args:
            embedding: L2-normalized 512-d vector
            box: [x1, y1, x2, y2]
            crop: BGR image (H, W, 3) or None
            score: detection confidence
            velocity: [vx, vy] or None
        """
        self.update_count += 1

        if embedding is not None:
            emb = _l2_norm(embedding.copy())
            self._update_embeddings(emb, score)

        self._update_body_features(box)

        if crop is not None:
            self._update_color_histogram(crop)

        if velocity is not None:
            self._update_motion(velocity)

    def _update_embeddings(self, emb, score):
        """Update EWMA embeddings and gallery."""
        if self.ewma_embedding is None:
            self.ewma_embedding = emb.copy()
            self.recent_embedding = emb.copy()
            self.best_embedding = emb.copy()
            self.best_score = score
            self.gallery = [emb.copy()]
            return

        # Confidence-weighted alpha
        conf_w = min(1.0, max(0.1, score))

        # Slow EWMA (stable identity signature)
        a_slow = self.EWMA_ALPHA_SLOW * conf_w
        self.ewma_embedding = _l2_norm(
            (1.0 - a_slow) * self.ewma_embedding + a_slow * emb)

        # Fast EWMA (recent appearance)
        a_fast = self.EWMA_ALPHA_FAST * conf_w
        self.recent_embedding = _l2_norm(
            (1.0 - a_fast) * self.recent_embedding + a_fast * emb)

        # Best quality
        if score > self.best_score:
            self.best_embedding = emb.copy()
            self.best_score = score

        # Diversity gallery
        is_diverse = all(
            float(np.dot(emb, g)) < self.GALLERY_DIVERSITY
            for g in self.gallery
        )
        if is_diverse:
            if len(self.gallery) >= self.MAX_GALLERY:
                self.gallery.pop(0)
            self.gallery.append(emb.copy())

    def _update_body_features(self, box):
        """Update body aspect ratio and size features (EWMA)."""
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ar = w / h

        alpha = 0.15
        if self.update_count <= 1:
            self.avg_aspect_ratio = ar
            self.avg_height = h
            self.avg_width = w
        else:
            self.avg_aspect_ratio = (1 - alpha) * self.avg_aspect_ratio + alpha * ar
            self.avg_height = (1 - alpha) * self.avg_height + alpha * h
            self.avg_width = (1 - alpha) * self.avg_width + alpha * w

    def _update_color_histogram(self, crop):
        """Extract upper/lower body color histograms from crop (CPU only)."""
        if crop is None or crop.size == 0:
            return
        h, w = crop.shape[:2]
        if h < 10 or w < 5:
            return

        mid = h // 2
        upper = crop[:mid, :, :]
        lower = crop[mid:, :, :]

        upper_hist = self._compute_hs_histogram(upper)
        lower_hist = self._compute_hs_histogram(lower)

        alpha = 0.2
        if self.upper_color_hist is None:
            self.upper_color_hist = upper_hist
            self.lower_color_hist = lower_hist
        else:
            self.upper_color_hist = (1 - alpha) * self.upper_color_hist + alpha * upper_hist
            self.lower_color_hist = (1 - alpha) * self.lower_color_hist + alpha * lower_hist

    def _compute_hs_histogram(self, region_bgr):
        """Compute normalized H-S histogram from a BGR image region."""
        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.H_BINS, self.S_BINS],
            [0, 180, 0, 256]
        )
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _update_motion(self, velocity):
        """Update motion profile from velocity vector."""
        vx, vy = float(velocity[0]), float(velocity[1])
        speed = math.hypot(vx, vy)
        direction = math.atan2(vy, vx)

        alpha = 0.2
        if self.update_count <= 1:
            self.avg_speed = speed
            self.avg_direction = direction
        else:
            self.avg_speed = (1 - alpha) * self.avg_speed + alpha * speed
            # Circular EWMA for direction
            diff = direction - self.avg_direction
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            self.avg_direction += alpha * diff

        # Speed variance
        self.velocity_history.append(speed)
        if len(self.velocity_history) > 20:
            self.velocity_history.pop(0)
        if len(self.velocity_history) >= 3:
            self.speed_variance = float(np.std(self.velocity_history))

    # ── COMPARISON ────────────────────────────────────────────────────

    def compare(self, query_embedding, query_box=None, query_velocity=None,
                query_crop=None):
        """
        Compute multi-signal identity score against a query.

        Returns float in [0, 1]. Higher = more likely same person.
        """
        if self.ewma_embedding is None or query_embedding is None:
            return 0.0

        emb = _l2_norm(query_embedding.copy())

        # 1. Appearance: EWMA embedding similarity
        appearance = float(np.dot(self.ewma_embedding, emb))
        appearance = max(0.0, appearance)

        # 2. Historical: best match across gallery + recent + best
        historical = self._gallery_max_sim(emb)

        # 3. Color histogram similarity
        color = 0.5  # neutral if no crop
        if query_crop is not None and self.upper_color_hist is not None:
            color = self._color_similarity(query_crop)

        # 4. Motion consistency
        motion = 0.5  # neutral if no velocity
        if query_velocity is not None:
            motion = self._motion_similarity(query_velocity)

        # 5. Body feature similarity
        body = 0.5  # neutral if no box
        if query_box is not None:
            body = self._body_similarity(query_box)

        # Weighted fusion
        score = (
            self.W_APPEARANCE * appearance +
            self.W_HISTORICAL * historical +
            self.W_COLOR * color +
            self.W_MOTION * motion +
            self.W_BODY * body
        )

        return float(min(1.0, max(0.0, score)))

    def _gallery_max_sim(self, emb):
        """Max cosine similarity against gallery + stable embeddings."""
        sims = []
        if self.ewma_embedding is not None:
            sims.append(float(np.dot(self.ewma_embedding, emb)))
        if self.recent_embedding is not None:
            sims.append(float(np.dot(self.recent_embedding, emb)))
        if self.best_embedding is not None:
            sims.append(float(np.dot(self.best_embedding, emb)))
        for g in self.gallery:
            sims.append(float(np.dot(g, emb)))
        return max(sims) if sims else 0.0

    def _color_similarity(self, query_crop):
        """Compare color histograms using Bhattacharyya distance."""
        if query_crop is None or query_crop.size == 0:
            return 0.5
        h, w = query_crop.shape[:2]
        if h < 10 or w < 5:
            return 0.5

        mid = h // 2
        q_upper = self._compute_hs_histogram(query_crop[:mid])
        q_lower = self._compute_hs_histogram(query_crop[mid:])

        sim_upper = 1.0 - cv2.compareHist(
            self.upper_color_hist, q_upper, cv2.HISTCMP_BHATTACHARYYA)
        sim_lower = 1.0 - cv2.compareHist(
            self.lower_color_hist, q_lower, cv2.HISTCMP_BHATTACHARYYA)

        return max(0.0, 0.5 * sim_upper + 0.5 * sim_lower)

    def _motion_similarity(self, query_velocity):
        """Compare motion profile (speed + direction)."""
        qvx, qvy = float(query_velocity[0]), float(query_velocity[1])
        q_speed = math.hypot(qvx, qvy)
        q_dir = math.atan2(qvy, qvx)

        # Speed similarity (normalized difference)
        max_speed = max(self.avg_speed, q_speed, 1.0)
        speed_sim = 1.0 - abs(self.avg_speed - q_speed) / max_speed

        # Direction similarity (cosine of angle difference)
        dir_diff = abs(q_dir - self.avg_direction)
        if dir_diff > math.pi:
            dir_diff = 2 * math.pi - dir_diff
        dir_sim = 0.5 * (1.0 + math.cos(dir_diff))

        # If both are near-stationary, direction is meaningless
        if self.avg_speed < 1.0 and q_speed < 1.0:
            return 0.8  # Slight positive bias for stationary people

        return 0.5 * speed_sim + 0.5 * dir_sim

    def _body_similarity(self, query_box):
        """Compare body features (aspect ratio, height)."""
        x1, y1, x2, y2 = query_box
        q_w = max(1, x2 - x1)
        q_h = max(1, y2 - y1)
        q_ar = q_w / q_h

        if self.avg_height < 1:
            return 0.5

        # Aspect ratio similarity
        ar_diff = abs(q_ar - self.avg_aspect_ratio)
        ar_sim = max(0.0, 1.0 - ar_diff * 3.0)  # Penalize >0.33 diff

        # Height similarity (scale-invariant-ish)
        h_ratio = min(q_h, self.avg_height) / max(q_h, self.avg_height, 1)
        h_sim = h_ratio  # 1.0 when same size, 0.5 when 2x different

        return 0.5 * ar_sim + 0.5 * h_sim

    # ── SERIALIZATION ─────────────────────────────────────────────────

    def to_dict(self):
        """Export fingerprint for warm memory storage."""
        return {
            "ewma_embedding": self.ewma_embedding,
            "recent_embedding": self.recent_embedding,
            "best_embedding": self.best_embedding,
            "best_score": self.best_score,
            "gallery": list(self.gallery),
            "upper_color_hist": self.upper_color_hist,
            "lower_color_hist": self.lower_color_hist,
            "avg_aspect_ratio": self.avg_aspect_ratio,
            "avg_height": self.avg_height,
            "avg_width": self.avg_width,
            "avg_speed": self.avg_speed,
            "avg_direction": self.avg_direction,
            "speed_variance": self.speed_variance,
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstruct fingerprint from warm memory data."""
        fp = cls()
        fp.ewma_embedding = data.get("ewma_embedding")
        fp.recent_embedding = data.get("recent_embedding")
        fp.best_embedding = data.get("best_embedding")
        fp.best_score = data.get("best_score", 0.0)
        fp.gallery = data.get("gallery", [])
        fp.upper_color_hist = data.get("upper_color_hist")
        fp.lower_color_hist = data.get("lower_color_hist")
        fp.avg_aspect_ratio = data.get("avg_aspect_ratio", 0.0)
        fp.avg_height = data.get("avg_height", 0.0)
        fp.avg_width = data.get("avg_width", 0.0)
        fp.avg_speed = data.get("avg_speed", 0.0)
        fp.avg_direction = data.get("avg_direction", 0.0)
        fp.speed_variance = data.get("speed_variance", 0.0)
        fp.update_count = data.get("update_count", 0)
        return fp
