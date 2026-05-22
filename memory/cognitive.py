"""
SELFWATCH — Cognitive Memory System v2

A human-inspired identity memory module for persistent object tracking.
This module is completely independent of any specific detector (RF-DETR, YOLO)
or tracker (StrongSORT, ByteTrack). It operates purely on mathematical
tracking outputs: embeddings, bounding boxes, confidence scores, and timestamps.

Architecture:
    The MEMORY defines identity. The tracker only provides observations.
    This module decides WHO someone is; the tracker decides WHERE they are.

Seven Core Systems:
    1. MEMORY LOCK          — Prevents identity corruption from bad detections
    2. WARM MEMORY          — Stores lost identities for future recovery
    3. COGNITIVE FORGETTING — Exponential decay simulating human-like forgetting
    4. ACTIVE RECALL        — Re-identifies returning people from warm memory
    5. MULTI-EMBEDDING      — Stable/recent/best-quality embedding triplet
    6. EMBEDDING DIVERSITY  — Gallery of diverse representative embeddings
    7. DYNAMIC DECAY        — Scene-adaptive memory persistence
"""

import time
import math
import numpy as np
from collections import deque


# ═════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════

# Memory Lock thresholds
EDGE_MARGIN_PX = 5
AREA_DROP_RATIO = 0.40          # Relaxed: Allow up to 40% partial occlusion
ASPECT_CHANGE_THRESH = 0.5
MIN_CONFIDENCE = 0.30           # Relaxed: RF-DETR outputs 0.3-0.4 in crowds
MIN_QUALITY_FOR_UPDATE = 0.35

# Embedding diversity
MAX_GALLERY_SIZE = 5
GALLERY_DIVERSITY_THRESH = 0.95   # New embedding must be < 0.95 sim to all

# Stability voting
STABILITY_FRAMES_REQUIRED = 3

# Identity states
STATE_ACTIVE = "ACTIVE"
STATE_UNCERTAIN = "UNCERTAIN"
STATE_LOST = "LOST"
STATE_WARM_MEMORY = "WARM_MEMORY"
STATE_ARCHIVED = "ARCHIVED"


# ═════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════

def _cosine_similarity(a, b):
    """Fast cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denom = norm_a * norm_b
    if denom < 1e-8:
        return 0.0
    return float(dot / denom)


def _l2_normalize(v):
    """L2-normalize a vector in-place safe copy."""
    v = v.copy().astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 1e-6:
        v /= norm
    return v


class CognitiveMemory:
    """
    The 'brain' of SELFWATCH — manages identity persistence across time.

    This class is tracker-agnostic and detector-agnostic. It receives only
    raw mathematical observations (embeddings, boxes, scores) and makes
    memory decisions independently.

    Systems:
        1. Memory Lock          — should_lock_memory()
        2. Warm Memory Storage  — save_lost_track()
        3. Cognitive Forgetting — update_and_decay()
        4. Active Recall        — retrieve_identity()
        5. Multi-Embedding      — update_active_identity()
        6. Embedding Diversity  — managed inside active identities
        7. Dynamic Decay        — set_scene_difficulty()

    Args:
        base_decay_rate:    Controls how fast memories fade (lower = slower).
        archive_threshold:  Confidence floor; below this, memories are deleted.
    """

    def __init__(self, base_decay_rate=0.05, archive_threshold=0.3,
                 max_warm=100, max_archive=500,
                 fusion_weights=None, fusion_threshold=0.78,
                 event_logger=None, metrics=None):
        # ── Warm Memory: the graveyard of lost identities ────────────
        self.warm_memory = {}

        # ── Archive: historical only, no retrieval ───────────────────
        self._archive = []
        self._max_archive = max_archive

        # ── Decay parameters ─────────────────────────────────────────
        self.base_decay_rate = base_decay_rate
        self.archive_threshold = archive_threshold
        self._difficulty_multiplier = 1.0

        # ── Hard limits ──────────────────────────────────────────────
        self._max_warm = max_warm

        # ── Confidence Fusion ────────────────────────────────────────
        self._fusion_weights = fusion_weights or [0.40, 0.10, 0.10, 0.15, 0.10, 0.15]
        self._fusion_threshold = fusion_threshold

        # ── Observability hooks (optional) ───────────────────────────
        self._logger = event_logger    # CognitiveEventLogger or None
        self._metrics = metrics        # TrackingMetrics or None

        # ── Per-track state (keyed by track_id) ──────────────────────
        self._track_history = {}
        self._identity_states = {}
        self._active_identities = {}
        self._stability_votes = {}
        self._consecutive_bad_frames = {}
        self._last_lock_reasons = {}

    # ═════════════════════════════════════════════════════════════════
    #  1. MEMORY LOCK — Prevents identity corruption
    # ═════════════════════════════════════════════════════════════════

    def should_lock_memory(self, track_id, bbox, confidence,
                           frame_dims=None):
        """
        Determines whether an embedding update should be FROZEN for a track.

        Protects stored visual identity from being overwritten by corrupted
        crops (partial views, blur, edge clipping, sudden occlusion).

        Args:
            track_id:      Unique track identifier.
            bbox:          [x1, y1, x2, y2] bounding box of the detection.
            confidence:    Detector confidence score (0.0 - 1.0).
            frame_dims:    (height, width) of the video frame, or None.

        Returns:
            (is_locked: bool, quality_score: float)
            - is_locked:     True if memory should be frozen this frame.
            - quality_score: 0.0–1.0 composite quality; use as EMA strength.
        """
        x1, y1, x2, y2 = bbox
        curr_w = max(x2 - x1, 1)
        curr_h = max(y2 - y1, 1)
        curr_area = curr_w * curr_h
        curr_aspect = curr_w / curr_h

        reasons = []
        quality_components = []

        # ── 1a. Frame Edge Check (Area-Percentage Truncation) ────────
        if frame_dims is not None:
            fh, fw = frame_dims[:2]
            # Calculate what percentage of bbox is visible inside the frame
            clipped_x1 = max(0, x1)
            clipped_y1 = max(0, y1)
            clipped_x2 = min(fw, x2)
            clipped_y2 = min(fh, y2)
            visible_area = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)
            visible_ratio = visible_area / max(curr_area, 1)

            # Only lock if >15% of person is outside frame
            if visible_ratio < 0.85:
                reasons.append("frame_edge")

            # Smooth edge quality based on visibility
            edge_quality = max(0.2, min(1.0, visible_ratio))
            quality_components.append(edge_quality)

        # ── 1b. Area Drop / Aspect Shift Detection ───────────────────
        history = self._track_history.get(track_id)
        if history is not None:
            prev_area = history["area"]
            prev_aspect = history["aspect"]

            if prev_area > 0 and curr_area < prev_area * (1.0 - AREA_DROP_RATIO):
                reasons.append("area_drop")

            if abs(curr_aspect - prev_aspect) > ASPECT_CHANGE_THRESH:
                reasons.append("aspect_shift")

            if prev_area > 0:
                area_ratio = min(curr_area, prev_area) / max(curr_area, prev_area)
            else:
                area_ratio = 1.0
            quality_components.append(area_ratio)

            aspect_diff = abs(curr_aspect - prev_aspect)
            aspect_quality = max(0.2, 1.0 - aspect_diff / ASPECT_CHANGE_THRESH)
            quality_components.append(aspect_quality)
        else:
            quality_components.append(1.0)
            quality_components.append(1.0)

        # ── 1c. Confidence Check ─────────────────────────────────────
        if confidence < MIN_CONFIDENCE:
            reasons.append("low_confidence")

        conf_quality = min(1.0, confidence / 0.9)
        quality_components.append(conf_quality)

        # ── Composite Quality Score ──────────────────────────────────
        quality_score = sum(quality_components) / len(quality_components)

        # ── 1d. Update history (Adaptive Area Adaptation) ────────────
        # Even if locked, we must allow the reference area to adapt slowly
        # to perspective changes (walking away) to avoid "Lock Loops".
        if track_id not in self._track_history:
            self._track_history[track_id] = {
                "area": curr_area,
                "aspect": curr_aspect,
                "confidence": confidence,
            }
        else:
            hist = self._track_history[track_id]
            # If NOT locked, update normally (fast)
            # If locked, update VERY slowly to allow eventual recovery from perspective shifts
            alpha = 0.9 if not reasons else 0.05 
            hist["area"] = (1.0 - alpha) * hist["area"] + alpha * curr_area
            hist["aspect"] = (1.0 - alpha) * hist["aspect"] + alpha * curr_aspect
            hist["confidence"] = (1.0 - alpha) * hist["confidence"] + alpha * confidence

        # ── Temporal Smoothing & Lock Logic ──────────────────────────
        self._last_lock_reasons[track_id] = reasons
        if reasons or quality_score < MIN_QUALITY_FOR_UPDATE:
            self._consecutive_bad_frames[track_id] = self._consecutive_bad_frames.get(track_id, 0) + 1
            self._identity_states[track_id] = STATE_UNCERTAIN
        else:
            self._consecutive_bad_frames[track_id] = 0
            self._identity_states[track_id] = STATE_ACTIVE

        bad_frames = self._consecutive_bad_frames.get(track_id, 0)
        
        is_locked = False
        lock_type = None

        if bad_frames >= 3:
            # Hard lock: Conditions persisted, freeze entirely
            is_locked = True
            lock_type = "hard_lock"
        elif bad_frames > 0:
            # Soft lock: Grace period, apply severe EMA penalty but don't freeze completely
            quality_score = quality_score * 0.1
            lock_type = "soft_lock"

        if lock_type:
            if self._logger:
                self._logger.log("memory_lock", track_id,
                                 reasons=reasons, 
                                 quality=round(quality_score, 3),
                                 lock_type=lock_type)
            if self._metrics:
                self._metrics.record_lock_event(reasons, lock_type)

        return is_locked, quality_score, lock_type

    def clear_track_history(self, track_id):
        """Remove all per-track state for a deleted track."""
        self._track_history.pop(track_id, None)
        self._identity_states.pop(track_id, None)
        self._active_identities.pop(track_id, None)
        self._stability_votes.pop(track_id, None)
        self._consecutive_bad_frames.pop(track_id, None)
        self._last_lock_reasons.pop(track_id, None)

    # ═════════════════════════════════════════════════════════════════
    #  2. ADAPTIVE EMA UPDATE — Multi-Embedding Identity Memory
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_update_alpha(quality_score):
        """
        Compute nonlinear EMA update strength from quality score.

        alpha = quality_score^2

        Bad frames (0.3 quality) → alpha 0.09 → barely touch memory.
        Good frames (0.9 quality) → alpha 0.81 → strong update.

        Returns:
            float: EMA alpha in [0.0, 1.0].
        """
        return min(1.0, max(0.0, quality_score ** 2))

    def update_active_identity(self, track_id, embedding, quality_score,
                               bbox=None):
        """
        Update the multi-embedding identity record for an active track.

        Maintains three embeddings per identity:
            - stable_embedding:       Slow EMA, resistant to noise.
            - recent_embedding:       Fast EMA, captures current appearance.
            - best_quality_embedding: Snapshot from highest-quality frame.

        Also maintains a diversity gallery of up to 5 representative
        embeddings from different poses/lighting/scales.

        Args:
            track_id:       Unique track identifier.
            embedding:      Raw 512-dim embedding from ReID extractor.
            quality_score:  Quality score from should_lock_memory().
            bbox:           Optional [x1,y1,x2,y2] for scale diversity.
        """
        emb = _l2_normalize(embedding)
        alpha = self.compute_update_alpha(quality_score)

        identity = self._active_identities.get(track_id)

        if identity is None:
            # First observation — initialize all three slots
            self._active_identities[track_id] = {
                "stable_embedding":       emb.copy(),
                "recent_embedding":       emb.copy(),
                "best_quality_embedding": emb.copy(),
                "best_quality_score":     quality_score,
                "gallery":                [emb.copy()],
                "update_count":           1,
            }
            return

        # ── Stable embedding: very slow EMA (alpha * 0.3) ────────────
        stable_alpha = alpha * 0.3
        identity["stable_embedding"] = _l2_normalize(
            (1.0 - stable_alpha) * identity["stable_embedding"]
            + stable_alpha * emb
        )

        # ── Recent embedding: fast EMA (alpha * 0.8) ─────────────────
        recent_alpha = alpha * 0.8
        identity["recent_embedding"] = _l2_normalize(
            (1.0 - recent_alpha) * identity["recent_embedding"]
            + recent_alpha * emb
        )

        # ── Best quality: replace only if this frame is better ────────
        if quality_score > identity["best_quality_score"]:
            identity["best_quality_embedding"] = emb.copy()
            identity["best_quality_score"] = quality_score

        # ── Diversity gallery: add if sufficiently different ──────────
        gallery = identity["gallery"]
        is_diverse = all(
            _cosine_similarity(emb, g) < GALLERY_DIVERSITY_THRESH
            for g in gallery
        )
        if is_diverse:
            if len(gallery) >= MAX_GALLERY_SIZE:
                # Replace the oldest entry
                gallery.pop(0)
            gallery.append(emb.copy())

        identity["update_count"] += 1

    def get_identity_embeddings(self, track_id):
        """
        Get all stored embeddings for an active track.

        Returns:
            dict with keys: stable, recent, best_quality, gallery.
            Or None if track has no identity record.
        """
        identity = self._active_identities.get(track_id)
        if identity is None:
            return None
        return {
            "stable":       identity["stable_embedding"],
            "recent":       identity["recent_embedding"],
            "best_quality": identity["best_quality_embedding"],
            "gallery":      list(identity["gallery"]),
        }

    # ═════════════════════════════════════════════════════════════════
    #  3. WARM MEMORY STORAGE — The graveyard of lost identities
    # ═════════════════════════════════════════════════════════════════

    def save_lost_track(self, track_id, final_embedding, duration_frames,
                        quality_score, last_position=None, velocity=None,
                        gait_signature=None, frame_index=None):
        """
        Stores a lost identity into warm memory for future retrieval.

        Called when a confirmed track is deleted by the tracker. The person
        disappears from the screen, but their visual identity is preserved
        here so we can recognize them if they return.

        Args:
            track_id:         The track's unique ID.
            final_embedding:  L2-normalized 512-dim embedding (np.ndarray).
            duration_frames:  How many frames this track existed.
            quality_score:    Average quality of their detections (0.0–1.0).
            last_position:    [x1, y1, x2, y2] last known bounding box.
            velocity:         [vx, vy] velocity vector (pixels/frame) or None.
        """
        importance = (duration_frames / 30.0) + quality_score

        # ── Collect multi-embeddings from active identity if available ─
        identity = self._active_identities.get(track_id)
        if identity is not None:
            stable_emb = identity["stable_embedding"].copy()
            recent_emb = identity["recent_embedding"].copy()
            best_emb = identity["best_quality_embedding"].copy()
            gallery = [g.copy() for g in identity["gallery"]]
        else:
            # Fallback: use the single provided embedding for all slots
            emb = _l2_normalize(final_embedding)
            stable_emb = emb.copy()
            recent_emb = emb.copy()
            best_emb = emb.copy()
            gallery = [emb.copy()]

        # ── Enforce hard memory limit ─────────────────────────────────
        if len(self.warm_memory) >= self._max_warm:
            # Prune the lowest-confidence entry
            worst_id = min(self.warm_memory,
                           key=lambda k: self.warm_memory[k]["confidence"])
            self._archive_entry(worst_id)
            if self._logger:
                self._logger.log("memory_prune", worst_id,
                                 reason="capacity_limit")
            if self._metrics:
                self._metrics.record_memory_prune()

        now_perf = time.perf_counter()
        self.warm_memory[track_id] = {
            # Multi-embedding triplet
            "stable_embedding":       stable_emb,
            "recent_embedding":       recent_emb,
            "best_quality_embedding": best_emb,
            "gallery":                gallery,
            # Legacy single embedding (backward compat)
            "embedding":              _l2_normalize(final_embedding),
            # Memory metadata
            "importance":    importance,
            "confidence":    1.0,
            "timestamp":     now_perf,
            "lost_timestamp": now_perf,
            "decay_timestamp": now_perf,
            "last_seen_frame": frame_index,
            "last_position": (np.array(last_position, dtype=np.float32)
                              if last_position is not None else None),
            "velocity":      (np.array(velocity, dtype=np.float32)
                              if velocity is not None else None),
            "last_seen":     time.time(),
            "gait_signature": (gait_signature.copy()
                              if gait_signature is not None else None),
        }

        # Update identity state
        self._identity_states[track_id] = STATE_WARM_MEMORY

        # Clean up active tracking data
        self._track_history.pop(track_id, None)
        self._active_identities.pop(track_id, None)
        self._stability_votes.pop(track_id, None)

        # Observability
        if self._logger:
            self._logger.log("memory_save", track_id,
                             importance=round(importance, 2),
                             gallery_size=len(gallery))
        if self._metrics:
            self._metrics.record_memory_save(track_id, duration_frames)

        print(f"[MEMORY] Track {track_id} saved to Warm Memory! "
              f"Importance: {importance:.2f}, "
              f"Gallery: {len(gallery)} embeddings")

    # ═════════════════════════════════════════════════════════════════
    #  4. COGNITIVE FORGETTING — Exponential memory decay
    # ═════════════════════════════════════════════════════════════════

    def update_and_decay(self):
        """
        Run every frame. Gradually decays warm memory entries over time.

        Uses importance-weighted exponential decay with scene difficulty
        adjustment (set via set_scene_difficulty):

            adjusted_decay  = base_decay_rate / difficulty_multiplier
            effective_decay = adjusted_decay / (1.0 + importance)
            confidence_t    = confidence_{t-1} * exp(-effective_decay * dt)

        Important memories decay slower. Harder scenes also slow decay.
        When confidence drops below the archive threshold, the memory is
        permanently deleted.
        """
        current_time = time.perf_counter()
        expired_keys = []

        # Apply dynamic difficulty adjustment
        adjusted_base = self.base_decay_rate / self._difficulty_multiplier

        for track_id, mem in self.warm_memory.items():
            last_decay = mem.get("decay_timestamp", mem.get("timestamp", current_time))
            dt = current_time - last_decay
            mem["decay_timestamp"] = current_time

            effective_decay = adjusted_base / (1.0 + mem["importance"])

            mem["confidence"] = mem["confidence"] * math.exp(
                -effective_decay * dt
            )

            if mem["confidence"] < self.archive_threshold:
                expired_keys.append(track_id)

        for track_id in expired_keys:
            survival = time.time() - self.warm_memory[track_id].get("last_seen", time.time())
            print(f"[DECAY] Track {track_id} faded from memory. "
                  f"(conf={self.warm_memory[track_id]['confidence']:.3f})")
            self._archive_entry(track_id)
            if self._logger:
                self._logger.log("memory_decay", track_id,
                                 survival_s=round(survival, 2))
            if self._metrics:
                self._metrics.record_memory_decay(track_id, survival)

    # ═════════════════════════════════════════════════════════════════
    #  5. ACTIVE RECALL — Identity resurrection from warm memory
    # ═════════════════════════════════════════════════════════════════

    def retrieve_identity(self, new_embedding, current_position,
                          current_time, threshold=0.85,
                          gait_signature=None, topology=None,
                          new_gait_sig=None, current_frame_index=None,
                          fps_estimate=5.0, new_velocity=None):
        """
        Searches warm memory for a visual match to a newly spawned track.

        Uses multi-embedding comparison: checks new embedding against the
        stable, recent, best-quality, AND gallery embeddings of each warm
        memory entry. The maximum similarity across all stored embeddings
        is used for the match decision.

        Velocity-aware: if the lost track had a known velocity, the method
        predicts where they should be and rejects physically implausible
        resurrections.

        Args:
            new_embedding:     L2-normalized 512-dim embedding.
            current_position:  [x1, y1, x2, y2] bounding box.
            current_time:      time.perf_counter() timestamp.
            threshold:         Minimum cosine similarity (default 0.85).

        Returns:
            int or None: The recovered track_id, or None.
        """
        if not self.warm_memory:
            return None

        emb = _l2_normalize(new_embedding)
        candidates = []  # (track_id, fusion_score, emb_sim, vel_score, conf)

        new_cx = (current_position[0] + current_position[2]) / 2.0
        new_cy = (current_position[1] + current_position[3]) / 2.0

        for track_id, mem in self.warm_memory.items():
            lost_time = mem.get("lost_timestamp", mem.get("timestamp", current_time))
            dt_s = current_time - lost_time
            lost_frame = mem.get("last_seen_frame")
            if current_frame_index is not None and lost_frame is not None:
                frames_elapsed = max(1.0, float(current_frame_index - lost_frame))
            else:
                frames_elapsed = max(1.0, dt_s * max(float(fps_estimate), 1.0))

            # ── Spatial plausibility ─────────────────────────────────
            if mem["last_position"] is not None:
                old_box = mem["last_position"]
                old_cx = (old_box[0] + old_box[2]) / 2.0
                old_cy = (old_box[1] + old_box[3]) / 2.0

                # ── Velocity-aware prediction ────────────────────────
                if mem["velocity"] is not None:
                    vel = mem["velocity"]
                    # vel is px/frame. Multiply by frames_elapsed.
                    pred_cx = old_cx + vel[0] * frames_elapsed
                    pred_cy = old_cy + vel[1] * frames_elapsed

                    # Distance from predicted position
                    pred_dist = math.hypot(new_cx - pred_cx, new_cy - pred_cy)
                    # Allow larger radius since prediction is uncertain
                    max_pred_dist = min(3000.0, 150.0 * frames_elapsed + 300.0)
                    if pred_dist > max_pred_dist:
                        continue
                else:
                    # No velocity — fall back to raw distance check
                    dist = math.hypot(new_cx - old_cx, new_cy - old_cy)
                    max_dist = min(3000.0, 250.0 * frames_elapsed + 200.0)
                    if dist > max_dist:
                        continue

                # ── Direction-consistency gate ────────────────────────
                if mem["velocity"] is not None and new_velocity is not None:
                    sv = np.array(mem["velocity"], dtype=np.float32)
                    nv = np.array(new_velocity, dtype=np.float32)
                    sv_speed = float(np.linalg.norm(sv))
                    nv_speed = float(np.linalg.norm(nv))
                    if sv_speed > 0.5 and nv_speed > 0.5:
                        dot_dir = float(np.dot(sv / sv_speed, nv / nv_speed))
                        if dot_dir < 0:
                            continue  # Opposite direction → physically impossible

            # ── Multi-embedding similarity (batched) ──────────────────
            all_embs = [mem["stable_embedding"], mem["recent_embedding"],
                        mem["best_quality_embedding"]]
            all_embs.extend(mem.get("gallery", []))
            if "embedding" in mem:
                all_embs.append(mem["embedding"])
            emb_matrix = np.stack(all_embs)  # (K, 512)
            sims = emb_matrix @ emb  # Single matrix-vector dot product
            emb_sim = float(np.max(sims))

            # ── Velocity plausibility score ──────────────────────────
            if mem["last_position"] is not None:
                old_box = mem["last_position"]
                old_cx = (old_box[0] + old_box[2]) / 2.0
                old_cy = (old_box[1] + old_box[3]) / 2.0
                dist = math.hypot(new_cx - old_cx, new_cy - old_cy)
                max_dist = min(3000.0, 250.0 * frames_elapsed + 200.0)
                vel_score = max(0.0, 1.0 - dist / max(max_dist, 1.0))
            else:
                vel_score = 0.5  # Unknown position, neutral

            # ── Confidence Fusion (6-weight) ────────────────────────────
            # Gait similarity
            gait_sim = 0.5  # Neutral default
            if gait_signature is not None and new_gait_sig is not None:
                stored_gait = mem.get("gait_signature")
                if stored_gait is not None:
                    from memory.gait import GaitSignature
                    gait_sim = GaitSignature.compare(new_gait_sig, stored_gait)

            # Topology spatial prior
            topo_prior = 0.5  # Neutral default
            if topology is not None and mem["last_position"] is not None:
                topo_prior = topology.get_spatial_prior(
                    mem["last_position"].tolist(), list(current_position))

            w = self._fusion_weights
            # 6-weight: emb + conf + quality + velocity + gait + topology
            quality_score = mem.get("best_quality_score", 0.7)
            fusion = (w[0] * emb_sim +
                      w[1] * mem["confidence"] +
                      w[2] * quality_score +
                      w[3] * vel_score +
                      w[4] * gait_sim +
                      w[5] * topo_prior)

            candidates.append((track_id, fusion, emb_sim, vel_score,
                                mem["confidence"], gait_sim, topo_prior))

        # ── Sort candidates by fusion score ──────────────────────────
        candidates.sort(key=lambda x: x[1], reverse=True)

        # ── Diagnostic: print best candidate for tuning ──────────────
        if candidates:
            c = candidates[0]
            status = "✓ PASS" if c[1] >= self._fusion_threshold else "✗ FAIL"
            print(f"[RETRIEVE] Best: ID {c[0]} fusion={c[1]:.3f} "
                  f"(emb={c[2]:.3f} vel={c[3]:.3f} conf={c[4]:.3f} "
                  f"gait={c[5]:.3f} topo={c[6]:.3f}) "
                  f"thresh={self._fusion_threshold:.3f} {status}")

        # ── Log top-3 candidates for future tuning ───────────────────
        if self._logger and candidates:
            top3 = [{"id": int(c[0]), "fusion": round(c[1], 3),
                      "emb_sim": round(c[2], 3), "vel": round(c[3], 3),
                      "conf": round(c[4], 3),
                      "gait": round(c[5], 3), "topo": round(c[6], 3)}
                     for c in candidates[:3]]
            self._logger.log("retrieval_candidates", candidates=top3)

        if self._metrics:
            self._metrics.record_retrieval_attempt(
                bool(candidates and candidates[0][1] >= self._fusion_threshold)
            )

        # ── Decision: resurrect or not ───────────────────────────────
        if candidates and candidates[0][1] >= self._fusion_threshold:
            best_id, fusion_score, emb_sim = candidates[0][:3]

            # Relative thresholding check
            if len(candidates) > 1:
                second_best_fusion = candidates[1][1]
                if (fusion_score - second_best_fusion) <= 0.15:
                    print(f"[RECALL] Ambiguous match for ID {best_id}. Margin too small: {fusion_score:.2f} vs {second_best_fusion:.2f}")
                    return None

            mem = self.warm_memory[best_id]
            gap = current_time - mem.get("lost_timestamp", mem.get("timestamp", current_time))

            print(f"[RECALL] ★ RESURRECTION! Restored ID {best_id} "
                  f"(Fusion: {fusion_score:.2f}, Sim: {emb_sim:.2f}, "
                  f"Gap: {gap:.1f}s)")

            mem["confidence"] = 1.0
            mem["importance"] += 0.5

            if self._logger:
                self._logger.log("resurrection", best_id,
                                 fusion=round(fusion_score, 3),
                                 emb_sim=round(emb_sim, 3),
                                 gap_s=round(gap, 2))
            if self._metrics:
                self._metrics.record_resurrection(best_id, emb_sim, gap)

            return best_id

        return None

    # ═════════════════════════════════════════════════════════════════
    #  6. DETECTION STABILITY VOTING
    # ═════════════════════════════════════════════════════════════════

    def vote_for_new_identity(self, track_id, embedding, position):
        """
        Buffer new-track observations before confirming a new identity.

        A brand-new identity is only confirmed after STABILITY_FRAMES_REQUIRED
        consecutive stable detections. This eliminates ghost tracks and
        floating head detections that die within 1-2 frames.

        Call this every frame for tracks that have not yet been confirmed
        as genuine new identities.

        Args:
            track_id:   The new track's ID.
            embedding:  Current frame's embedding.
            position:   [x1, y1, x2, y2] current bounding box.

        Returns:
            bool: True if the identity is now confirmed (stable enough).
                  False if still accumulating votes.
        """
        vote = self._stability_votes.get(track_id)

        if vote is None:
            self._stability_votes[track_id] = {
                "count": 1,
                "last_position": np.array(position, dtype=np.float32),
                "first_embedding": _l2_normalize(embedding),
            }
            return False

        # Check spatial continuity: box shouldn't teleport between frames
        old_pos = vote["last_position"]
        old_cx = (old_pos[0] + old_pos[2]) / 2.0
        old_cy = (old_pos[1] + old_pos[3]) / 2.0
        new_cx = (position[0] + position[2]) / 2.0
        new_cy = (position[1] + position[3]) / 2.0
        frame_dist = math.hypot(new_cx - old_cx, new_cy - old_cy)

        # If the box jumped more than 150px between frames, reset votes
        if frame_dist > 150.0:
            self._stability_votes[track_id] = {
                "count": 1,
                "last_position": np.array(position, dtype=np.float32),
                "first_embedding": _l2_normalize(embedding),
            }
            return False

        # Check embedding consistency with first observation
        emb = _l2_normalize(embedding)
        sim = _cosine_similarity(emb, vote["first_embedding"])
        if sim < 0.6:
            # Embedding drifted too much — probably not the same person
            self._stability_votes[track_id] = {
                "count": 1,
                "last_position": np.array(position, dtype=np.float32),
                "first_embedding": emb,
            }
            return False

        vote["count"] += 1
        vote["last_position"] = np.array(position, dtype=np.float32)

        if vote["count"] >= STABILITY_FRAMES_REQUIRED:
            # Confirmed! Clean up vote buffer
            del self._stability_votes[track_id]
            self._identity_states[track_id] = STATE_ACTIVE
            return True

        return False

    # ═════════════════════════════════════════════════════════════════
    #  7. DYNAMIC MEMORY DECAY — Scene difficulty adjustment
    # ═════════════════════════════════════════════════════════════════

    def set_scene_difficulty(self, n_tracks, avg_confidence):
        """
        Adjust memory decay rate based on current scene difficulty.

        Call once per frame with current scene stats. Harder scenes
        (more people, lower confidence) slow down memory decay so
        important identities persist longer during challenging conditions.

        Args:
            n_tracks:        Number of currently tracked people.
            avg_confidence:  Average detection confidence this frame (0-1).
        """
        # More people = harder scene = slower decay
        crowd_factor = min(2.0, 1.0 + n_tracks * 0.15)

        # Lower confidence = harder scene = slower decay
        conf_factor = min(2.0, 1.0 + max(0.0, 1.0 - avg_confidence))

        # Combined: harder scenes → higher multiplier → slower decay
        self._difficulty_multiplier = crowd_factor * conf_factor

    # ═════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ═════════════════════════════════════════════════════════════════

    def _archive_entry(self, track_id):
        """Move a warm memory entry to the archive layer (historical only)."""
        mem = self.warm_memory.pop(track_id, None)
        if mem is None:
            return
        self._identity_states[track_id] = STATE_ARCHIVED
        self._archive.append({
            "track_id": int(track_id),
            "importance": mem.get("importance", 0),
            "archived_at": time.time(),
        })
        # Cap archive size (FIFO)
        while len(self._archive) > self._max_archive:
            self._archive.pop(0)

    @property
    def warm_count(self):
        """Number of identities currently in warm memory."""
        return len(self.warm_memory)

    @property
    def archive_count(self):
        """Number of entries in the archive."""
        return len(self._archive)

    def get_identity_state(self, track_id):
        """Get the current state of an identity."""
        return self._identity_states.get(track_id, None)

    def get_debug_info(self, track_id):
        """
        Returns all debug-relevant state for a track.

        Used by DebugOverlay to render per-track annotations.
        """
        history = self._track_history.get(track_id)
        identity = self._active_identities.get(track_id)

        return {
            "state": self._identity_states.get(track_id, "UNKNOWN"),
            "quality": round(history["confidence"], 2) if history else 0.0,
            "locked": self._identity_states.get(track_id) == STATE_UNCERTAIN,
            "lock_reasons": self._last_lock_reasons.get(track_id, []),
            "gallery_size": len(identity["gallery"]) if identity else 0,
        }

    def remove_from_warm(self, track_id):
        """Remove a specific identity from warm memory (after resurrection)."""
        return self.warm_memory.pop(track_id, None)

    def __repr__(self):
        active_count = sum(
            1 for s in self._identity_states.values() if s == STATE_ACTIVE
        )
        uncertain_count = sum(
            1 for s in self._identity_states.values() if s == STATE_UNCERTAIN
        )
        return (f"CognitiveMemory(active={active_count}, "
                f"uncertain={uncertain_count}, "
                f"warm={self.warm_count}, "
                f"archive={self.archive_count}, "
                f"decay={self.base_decay_rate}, "
                f"difficulty={self._difficulty_multiplier:.2f})")
