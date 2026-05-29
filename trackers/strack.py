"""
STrack: Single object track for StrongSORT pipeline.

Features:
  - local_id: IMMUTABLE tracker-assigned ID (never changes)
  - global_id: display/identity ID (managed ONLY by GlobalIDMapper)
  - Linear velocity prediction for position estimation
  - Embedding history buffer (last N good embeddings) with averaged output
  - MEMORY LOCK: Multi-criteria quality gating for embedding updates:
      * Minimum detection confidence
      * Frame edge proximity (partial visibility)
      * Area/ratio spike detection
      * Per-frame embedding_quality_score (0.0 - 1.0)
      * memory_update_allowed flag
  - Embedding freeze during occlusion / lost state
  - Track lifecycle: tentative -> confirmed -> lost -> removed
"""

import numpy as np
import cv2
from collections import deque


# ── Memory Lock thresholds ───────────────────────────────────────────
MIN_CROP_AREA = 1500              # px^2: reject tiny crops
MIN_ASPECT_RATIO = 0.15           # w/h minimum
MAX_ASPECT_RATIO = 3.5            # w/h maximum
BBOX_SHRINK_RATIO = 0.45          # Flag if area drops to <45% of stable size
ASPECT_CHANGE_THRESH = 0.5        # Flag if aspect ratio changes by >50%
FRAME_EDGE_MARGIN = 8             # px: flag if bbox within 8px of frame edge


class STrack:
    """
    Single object track with Memory Lock protection for StrongSORT.

    Identity architecture:
      - local_id:  Immutable, tracker-assigned. Used internally for matching.
      - global_id: Display/identity ID. Managed EXCLUSIVELY by GlobalIDMapper.
                   Pipeline code must NEVER write to global_id directly.
      - id:        Property alias that returns global_id for backward compat.

    The Memory Lock system prevents identity corruption by freezing
    embedding updates when detection quality is unreliable, while
    allowing spatial tracking to continue normally.
    """
    _next_id = 1

    # ── Track states ─────────────────────────────────────────────────
    STATE_TENTATIVE = 0
    STATE_CONFIRMED = 1
    STATE_LOST = 2

    def __init__(self, box, score, embedding=None, label="object",
                 confirm_threshold=3, embedding_history_size=10):
        # IMMUTABLE local tracker ID
        self.local_id = STrack._next_id
        STrack._next_id += 1

        # Global display ID — starts equal to local_id.
        # ONLY GlobalIDMapper may change this.
        self.global_id = self.local_id

        self.box = np.array(box, dtype=np.float32)
        self.smooth_box = np.array(box, dtype=np.float32)
        self.score = score
        self.label = label

        # Velocity model
        self.vel = np.zeros(2, dtype=np.float32)

        # OC-SORT: last valid observation for coasting
        self._last_observed_box = np.array(box, dtype=np.float32)
        self._last_observed_vel = np.zeros(2, dtype=np.float32)
        self._frames_since_observation = 0

        # Forensic tracking info
        self.last_assoc_cost = 0.0
        self.last_assoc_method = "NEW"
        self.cbiou_buffer = 0

        # Lifecycle
        self.state = self.STATE_TENTATIVE
        self.age = 1
        self.total_hits = 1
        self.consecutive_hits = 1
        self.time_since_update = 0
        self.confirm_threshold = confirm_threshold

        # ── Embedding history ────────────────────────────────────────
        self._emb_history_size = embedding_history_size
        self._emb_history = deque(maxlen=embedding_history_size)
        self.embedding = None

        if embedding is not None:
            emb = embedding.copy()
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb /= norm
            self._emb_history.append(emb)
            self.embedding = emb.copy()

        # ── Memory Lock state ────────────────────────────────────────
        self.memory_update_allowed = True
        self.embedding_quality_score = 1.0
        self._freeze_reasons = []       # Debug: reasons for current freeze
        self._consecutive_bad_frames = 0


        # ── Stable reference values (for change detection) ───────────
        self._stable_area = self._box_area(box)
        self._stable_aspect = self._box_aspect(box)
        self._stable_conf = score
        self._stable_update_count = 0    # How many good updates we've had
        self.recovery_count = 0


    # ── ID property (backward compat) ────────────────────────────────

    @property
    def id(self):
        """Global display ID. Use global_id directly for clarity."""
        return self.global_id

    @id.setter
    def id(self, value):
        """
        DEPRECATED setter — only exists for backward compatibility.
        New code should NEVER call track.id = X.
        Use GlobalIDMapper instead.
        """
        self.global_id = value

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_confirmed(self):
        return self.state == self.STATE_CONFIRMED

    @property
    def is_tentative(self):
        return self.state == self.STATE_TENTATIVE

    @property
    def is_lost(self):
        return self.state == self.STATE_LOST

    @property
    def predicted_box(self):
        return getattr(self, '_pred_box', self.smooth_box)

    @property
    def has_embedding(self):
        return self.embedding is not None and len(self._emb_history) > 0

    # ── Geometry helpers ─────────────────────────────────────────────

    @staticmethod
    def _box_area(box):
        return max(0, (box[2] - box[0]) * (box[3] - box[1]))

    @staticmethod
    def _box_aspect(box):
        h = box[3] - box[1]
        w = box[2] - box[0]
        return (w / h) if h > 1 else 1.0

    # ── Averaged embedding ───────────────────────────────────────────

    def get_averaged_embedding(self):
        """Returns the EMA-smoothed embedding."""
        return self.embedding

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, frame_delta=1):
        """Predict next position using frame-gap-aware velocity model.
        
        OC-SORT observation-centric coasting:
          - For recent observations (<=5 frames): use RAW last detection box
            (not smooth_box, which lags during fast motion)
          - For extended loss (>5 frames): coast from LAST VALID OBSERVATION
            to prevent Kalman-style drift accumulation
        """
        frame_delta = max(1, int(frame_delta))

        # OC-SORT: choose prediction base
        if self._frames_since_observation > 5:
            # Coast from last observation — avoids compounding prediction error
            base_box = self._last_observed_box
            base_vel = self._last_observed_vel
        else:
            # Use RAW detection box (not smooth_box) — smooth_box lags behind
            # during fast motion, causing the predicted box to trail the person
            # and IoU to drop to zero, breaking association.
            base_box = self.box
            base_vel = self.vel

        cx = (base_box[0] + base_box[2]) / 2 + base_vel[0] * frame_delta
        cy = (base_box[1] + base_box[3]) / 2 + base_vel[1] * frame_delta
        w = base_box[2] - base_box[0]
        h = base_box[3] - base_box[1]

        self._pred_box = np.array([cx - w / 2, cy - h / 2,
                                   cx + w / 2, cy + h / 2],
                                  dtype=np.float32)
        return self._pred_box

    # ── Memory Lock: Triple-Threat ───────────────────────────────────

    def should_lock_memory(self, box, confidence, frame_shape=None):
        """
        Ultimate Memory Lock: Checks three things instantly.
        1. Frame Edges
        2. Area/Ratio Spikes
        3. Confidence Dips
        """
        x1, y1, x2, y2 = box
        curr_w, curr_h = x2 - x1, max(y2 - y1, 1) # Prevent div by zero
        curr_area = curr_w * curr_h
        curr_ratio = curr_w / curr_h
        
        reasons = []

        # 1. Edge Lock (Only if >15% of bbox is outside frame)
        if frame_shape is not None:
            frame_h, frame_w = frame_shape[:2]
            clipped_x1 = max(0, x1)
            clipped_y1 = max(0, y1)
            clipped_x2 = min(frame_w, x2)
            clipped_y2 = min(frame_h, y2)
            visible_area = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)
            visible_ratio = visible_area / max(curr_area, 1)
            if visible_ratio < 0.85:
                reasons.append("frame_edge")
                
        # 2. Geometry Lock (Did the box suddenly shrink or deform?)
        if hasattr(self, 'last_area'):
            area_drop = curr_area < (self.last_area * 0.50) 
            ratio_shift = abs(curr_ratio - self.last_ratio) > 0.5
            if area_drop:
                reasons.append("area_drop")
            if ratio_shift:
                reasons.append("ratio_shift")
                
        # 3. Confidence Lock (Quality-gated EWMA requires conf > 0.7)
        if confidence < 0.7: 
            reasons.append("low_conf")
            
        self._freeze_reasons = reasons

        if len(reasons) > 0:
            self._consecutive_bad_frames += 1
        else:
            self._consecutive_bad_frames = 0

        # CLEAR: Safe to update OSNet embedding
        if len(reasons) == 0:
            self.last_area = curr_area
            self.last_ratio = curr_ratio

        # Determine lock status (True = hard lock)
        if self._consecutive_bad_frames >= 3:
            return True, False
        elif self._consecutive_bad_frames > 0:
            return False, True
        else:
            return False, False

    # ── Update ───────────────────────────────────────────────────────

    def update(self, box, score, embedding=None,
              crop=None, box_alpha=0.5, min_quality_score=0.4,
              frame_shape=None, sibling_boxes=None, frame_delta=1):
        """
        Update track with a matched detection.

        Tracking (box, velocity) always updates.
        Embedding memory ONLY updates if Memory Lock allows it.
        """
        frame_delta = max(1, int(frame_delta))
        new_box = np.array(box, dtype=np.float32)

        # ── Always update spatial tracking ───────────────────────────
        nc = (new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2
        oc = (self._last_observed_box[0] + self._last_observed_box[2]) / 2, \
             (self._last_observed_box[1] + self._last_observed_box[3]) / 2
        observed_vel = np.array(
            [(nc[0] - oc[0]) / frame_delta, (nc[1] - oc[1]) / frame_delta],
            dtype=np.float32,
        )

        # Adaptive velocity smoothing: respond faster during fast motion
        # to prevent the predicted box from trailing behind the person.
        # α=0.15 for stationary/slow (smooth), α=0.5 for fast (responsive).
        obs_speed = float(np.linalg.norm(observed_vel))
        if obs_speed > 8.0:
            vel_alpha = 0.5    # Fast motion: respond quickly
        elif obs_speed > 3.0:
            vel_alpha = 0.3    # Moderate motion: balanced
        else:
            vel_alpha = 0.15   # Slow/stationary: smooth
        self.vel = (1.0 - vel_alpha) * self.vel + vel_alpha * observed_vel
        self.box = new_box

        # OC-SORT observation-centric re-update:
        # After extended loss, the smooth_box may have drifted significantly.
        # Snap it back toward the new observation more aggressively.
        if self._frames_since_observation > 5:
            # Recovery from extended occlusion: aggressive blend
            effective_alpha = min(0.85, box_alpha + 0.3)
        else:
            # Adaptive smooth_box: track fast-moving people more tightly
            # to prevent the prediction base from lagging behind
            if obs_speed > 8.0:
                effective_alpha = min(0.9, box_alpha + 0.35)   # Very tight tracking
            elif obs_speed > 3.0:
                effective_alpha = min(0.8, box_alpha + 0.2)    # Tighter tracking
            else:
                effective_alpha = box_alpha                     # Default smooth
        self.smooth_box = effective_alpha * new_box + (1 - effective_alpha) * self.smooth_box
        self.score = score

        # OC-SORT: save observation snapshot for future coasting
        self._last_observed_box = new_box.copy()
        self._last_observed_vel = self.vel.copy()
        self._frames_since_observation = 0

        # ── Memory Lock: compute quality and decide ──────────────────
        is_hard_locked, is_soft_locked = self.should_lock_memory(box, score, frame_shape)

        if embedding is not None and not self.is_lost:
            if not is_hard_locked:
                # Quality is good (or soft-locked) -- update embedding memory
                emb = embedding.copy()
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb /= norm
                    
                # Quality-gated EWMA update: 0.8 * old + 0.2 * new
                if self.embedding is None:
                    self.embedding = emb
                else:
                    self.embedding = 0.8 * self.embedding + 0.2 * emb
                    norm = np.linalg.norm(self.embedding)
                    if norm > 1e-6:
                        self.embedding /= norm
                
                self._emb_history.append(self.embedding.copy())
                self.memory_update_allowed = True
            else:
                # Quality too low -- FREEZE embedding memory
                self.memory_update_allowed = False
        else:
            self.memory_update_allowed = False

        # ── Lifecycle ────────────────────────────────────────────────
        self.consecutive_hits += 1
        self.total_hits += 1
        self.time_since_update = 0
        self.age += frame_delta

        if self.state == self.STATE_TENTATIVE and \
           self.total_hits >= self.confirm_threshold:
            self.state = self.STATE_CONFIRMED


        if self.state == self.STATE_LOST:
            self.state = self.STATE_CONFIRMED
            self.recovery_count += 1
            print(f"  [TRACKER DEBUG] STrack local={self.local_id} RECOVERED! recovery_count={self.recovery_count}")


    # ── Lost / removal ───────────────────────────────────────────────

    def mark_lost(self, frame_delta=1):
        """Mark track as lost (no match this frame)."""
        frame_delta = max(1, int(frame_delta))
        self.time_since_update += frame_delta
        self._frames_since_observation += frame_delta
        self.consecutive_hits = 0
        self.age += frame_delta
        self.memory_update_allowed = False
        if self.state == self.STATE_CONFIRMED:
            self.state = self.STATE_LOST
            print(f"[INSTRUMENTATION] TRACK ENTERED LOST STATE: local={self.local_id} time_since_update={self.time_since_update} age={self.age} hits={self.total_hits}")

    def should_remove(self, max_lost):
        """Check if this track should be permanently removed.
        
        Tentative tracks get an extended survival window that scales
        with accumulated evidence (total_hits).
        """
        if self.state == self.STATE_TENTATIVE:
            tentative_max = 30 + self.total_hits * 10
            if self.time_since_update > tentative_max:
                return True
        elif self.time_since_update > max_lost:
            return True
        return False

    def __repr__(self):
        state_name = {0: 'TENT', 1: 'CONF', 2: 'LOST'}
        lock = ' LOCKED' if not self.memory_update_allowed else ''
        return (f"STrack(local={self.local_id}, global={self.global_id}, "
                f"state={state_name.get(self.state)}, "
                f"hits={self.consecutive_hits}/{self.total_hits}, "
                f"lost={self.time_since_update}, "
                f"embs={len(self._emb_history)}{lock})")
