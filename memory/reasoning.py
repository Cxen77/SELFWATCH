"""
Layer 5: Cognitive Reasoning — Fingerprint-Enhanced Identity Resolution

Identity recovery uses multi-signal fingerprint scoring:
  1. THINKING identities: fingerprint.compare() with all signals
  2. Phantom tracker: embedding + spatial matching
  3. Warm memory: multi-embedding retrieval

CRITICAL RULES:
  - Each match is CONSUMED (one-to-one exclusivity)
  - Only young tracks (age <= 30) get memory assistance
  - All proposals are advisory — never mutates tracker state
"""

import numpy as np
import math


def _l2_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


class CognitiveReasoning:
    def __init__(self, retrieval_threshold=0.82):
        self.retrieval_threshold = retrieval_threshold

    def evaluate_tracks(self, tracks, phantom_tracker, active_memory, warm_memory,
                        thinking_identities, current_time, frame_delta=1,
                        metrics=None, restricted_gids=None):
        """
        Returns list of proposals:
            (local_id, proposed_global_id, confidence, source)
        """
        proposals = []
        claimed_phantoms = set()
        claimed_thinking = set()
        restricted_gids = restricted_gids or set()

        # Filter out THINKING identities that are currently frozen or in cooldown
        # This prevents generating proposals that would just get rejected anyway
        filtered_thinking = {
            gid: data for gid, data in thinking_identities.items()
            if gid not in restricted_gids
        }

        # Sort youngest first — most likely to be returning people
        sorted_tracks = sorted(tracks, key=lambda t: t.age)

        for track in sorted_tracks:
            if track.embedding is None:
                continue

            # Only young tracks get memory assistance
            if track.age > 30:
                continue

            emb = track.embedding
            box = track.smooth_box
            vel = track.vel if hasattr(track, 'vel') else None

            # ── Priority 1: Fingerprint match against THINKING identities ─
            if filtered_thinking:
                result = self._match_thinking_fingerprint(
                    emb, box, vel, filtered_thinking,
                    active_memory, claimed_thinking)
                if result:
                    gid, conf = result
                    claimed_thinking.add(gid)
                    if metrics:
                        metrics.record_retrieval_attempt(success=True)
                    proposals.append((
                        track.local_id, gid, conf, "thinking"
                    ))
                    continue

            # ── Priority 2: Phantom tracker ──────────────────────────────
            match = phantom_tracker.try_match(emb, box, vel)
            if match and match.track_id not in claimed_phantoms:
                claimed_phantoms.add(match.track_id)
                if metrics:
                    metrics.record_retrieval_attempt(success=True)
                proposals.append((
                    track.local_id, match.track_id, 0.95, "phantom"
                ))
                continue

            # ── Priority 3: Warm memory ──────────────────────────────────
            if warm_memory.count > 0:
                if metrics:
                    metrics.record_retrieval_attempt(success=False)
                result = self._search_warm_memory(
                    emb, box, warm_memory, current_time, frame_delta)
                if result:
                    proposed_gid, match_conf = result
                    if metrics:
                        metrics.retrieval_successes += 1
                    proposals.append((
                        track.local_id, proposed_gid, match_conf, "warm"
                    ))

        return proposals

    def _match_thinking_fingerprint(self, embedding, box, velocity,
                                     thinking_identities, active_memory,
                                     already_claimed):
        """
        Match a new track against THINKING identities using full
        fingerprint scoring (embedding + color + body + motion).

        Includes hard direction-consistency gating: if the new track's
        velocity opposes the THINKING identity's last known velocity,
        the match is rejected to prevent identity teleportation.
        """
        emb = _l2_norm(embedding.copy())
        best_gid = None
        best_score = 0.0

        # Pre-compute new detection velocity info
        new_vel = None
        new_speed = 0.0
        if velocity is not None:
            new_vel = np.array(velocity, dtype=np.float32)
            new_speed = float(np.linalg.norm(new_vel))

        for gid, data in thinking_identities.items():
            if gid in already_claimed:
                continue

            # ── Direction-consistency gate (applies to ALL paths) ─────
            stored_vel = data.get("last_velocity")
            if stored_vel is not None and new_vel is not None:
                sv = np.array(stored_vel, dtype=np.float32)
                sv_speed = float(np.linalg.norm(sv))
                if sv_speed > 0.5 and new_speed > 0.5:
                    dot_dir = float(np.dot(sv / sv_speed, new_vel / new_speed))
                    if dot_dir < 0:
                        # Opposite direction → reject: physically impossible
                        continue

            # Get fingerprint from active memory
            fp = active_memory.get_fingerprint(gid)
            if fp is None:
                # Fallback to raw embedding matching
                last_emb = data.get("last_embedding")
                last_box = data.get("last_box")
                if last_emb is None:
                    continue

                sim = float(np.dot(_l2_norm(last_emb), emb))
                if sim < 0.80:
                    continue

                # Spatial check
                if last_box is not None:
                    new_cx = (box[0] + box[2]) / 2.0
                    new_cy = (box[1] + box[3]) / 2.0
                    old_cx = (last_box[0] + last_box[2]) / 2.0
                    old_cy = (last_box[1] + last_box[3]) / 2.0
                    if math.hypot(new_cx - old_cx, new_cy - old_cy) > 300:
                        continue

                if sim > best_score:
                    best_score = sim
                    best_gid = gid
                continue

            # Full fingerprint comparison
            # Hard gate: raw EWMA embedding similarity must be reasonably plausible
            if fp.ewma_embedding is not None:
                raw_sim = float(np.dot(fp.ewma_embedding, emb))
                if raw_sim < 0.70:
                    continue  # Embedding too different, skip

            vel_list = velocity.tolist() if velocity is not None else None
            score = fp.compare(
                emb, query_box=box.tolist(),
                query_velocity=vel_list)

            # Spatial plausibility check
            last_box = data.get("last_box")
            if last_box is not None:
                new_cx = (box[0] + box[2]) / 2.0
                new_cy = (box[1] + box[3]) / 2.0
                old_cx = (last_box[0] + last_box[2]) / 2.0
                old_cy = (last_box[1] + last_box[3]) / 2.0
                dist = math.hypot(new_cx - old_cx, new_cy - old_cy)
                if dist > 300:
                    continue
                # Spatial bonus for proximity: trust motion/path continuity
                score += max(0.0, 1.0 - dist / 300.0) * 0.15
                
            # Reuse existing identity bonus
            score += 0.10

            if score > 0.65 and score > best_score:
                best_score = score
                best_gid = gid

        if best_gid is not None:
            return best_gid, min(1.0, best_score)
        return None

    def _search_warm_memory(self, embedding, box, warm_memory,
                            current_time, frame_delta):
        """Multi-embedding search over warm memory."""
        candidates = []
        new_cx = (box[0] + box[2]) / 2.0
        new_cy = (box[1] + box[3]) / 2.0

        for gid, mem in warm_memory.get_all().items():
            dt_s = max(0.1, current_time - mem["lost_time"])

            if mem["last_box"] is not None:
                old_box = mem["last_box"]
                old_cx = (old_box[0] + old_box[2]) / 2.0
                old_cy = (old_box[1] + old_box[3]) / 2.0

                if mem["velocity"] is not None:
                    vel = mem["velocity"]
                    vx = float(vel[0]) if hasattr(vel, '__len__') else 0.0
                    vy = float(vel[1]) if hasattr(vel, '__len__') else 0.0
                    frames_elapsed = dt_s * 15.0
                    pred_cx = old_cx + vx * frames_elapsed
                    pred_cy = old_cy + vy * frames_elapsed
                    pred_dist = math.hypot(new_cx - pred_cx, new_cy - pred_cy)
                    if pred_dist > min(3000.0, 150.0 * frames_elapsed + 300.0):
                        continue
                else:
                    dist = math.hypot(new_cx - old_cx, new_cy - old_cy)
                    if dist > min(3000.0, 250.0 * (dt_s * 15) + 200.0):
                        continue

            embs = mem["embeddings"]
            all_embs = []
            for key in ("stable", "recent", "best"):
                if embs[key] is not None:
                    all_embs.append(embs[key])
            all_embs.extend(embs["gallery"])

            if not all_embs:
                continue

            emb_matrix = np.stack(all_embs)
            sims = emb_matrix @ embedding
            emb_sim = float(np.max(sims))
            fusion = 0.8 * emb_sim + 0.2 * mem["decayed_confidence"]
            candidates.append((gid, fusion))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_gid, best_score = candidates[0]

        if best_score >= self.retrieval_threshold:
            if len(candidates) > 1 and (best_score - candidates[1][1]) < 0.10:
                return None
            return best_gid, best_score
        return None
