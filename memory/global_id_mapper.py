"""
SELFWATCH — Global ID Mapper (Single-Camera)

The ONLY authority for mapping local tracker IDs to persistent global IDs.
Advisory architecture: reads tracker state, NEVER mutates it.

Design rules:
  1. NEVER rewrite track.id or any tracker internal state.
  2. Maintains a local_id → global_id mapping table.
  3. Requires 5 consecutive stable frames before allowing any global ID change.
  4. All downstream rendering uses global_id from this mapper.
  5. Embedding similarity is the only signal (no gait, topology, etc.).
"""

import numpy as np
import time


def _cosine_sim(a, b):
    """Fast cosine similarity between two L2-normalized vectors."""
    dot = float(np.dot(a, b))
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


class GlobalIDMapper:
    """
    Maps tracker-local IDs to stable global display IDs.

    Architecture:
        - Each new local_id gets a unique global_id on first sight.
        - The active memory stores (global_id → embedding) for confirmed tracks.
        - When a new track appears, we compare its embedding against ALL
          stored global identities. If the similarity is very high AND the
          cooldown has elapsed, we can re-assign the global_id.
        - Cooldown: a global_id change requires MIN_STABLE_FRAMES consecutive
          frames where the same suggestion is made.

    This module is PASSIVE — it only provides a lookup table.
    The pipeline reads mapper.get_global_id(local_id) for rendering.
    """

    MIN_STABLE_FRAMES = 5          # Cooldown before allowing reassignment
    MATCH_THRESHOLD = 0.82         # Cosine similarity to accept identity match
    AMBIGUITY_MARGIN = 0.10        # Gap between 1st and 2nd best must exceed this

    def __init__(self):
        # local_id → global_id (the authoritative mapping)
        self._local_to_global = {}

        # global_id → { embedding, last_box }
        self._global_gallery = {}

        # Cooldown state: local_id → { suggested_gid, stable_count }
        self._reassign_proposals = {}

        # Counter for generating new unique global IDs
        self._next_global_id = 1

        # Tracks currently active (local_ids seen this frame)
        self._active_local_ids = set()

    # ── Public API ──────────────────────────────────────────────────

    def get_global_id(self, local_id):
        """Get the global display ID for a tracker-local ID."""
        return self._local_to_global.get(local_id, local_id)

    def update(self, tracks, reid_frame=False):
        """
        Called once per frame AFTER tracker.update().

        Reads track state (embedding, box, lifecycle) to:
          1. Register new tracks with fresh global IDs.
          2. Update gallery embeddings for confirmed tracks.
          3. Propose re-identification for young tracks (with cooldown).

        Args:
            tracks:      List of STrack objects from the tracker.
            reid_frame:  True if ReID was run this frame (embeddings are fresh).
        """
        self._active_local_ids = set()

        for track in tracks:
            lid = track.local_id
            self._active_local_ids.add(lid)

            # ── First time seeing this local_id ──────────────────────
            if lid not in self._local_to_global:
                # Try to match against gallery of lost global identities
                suggested_gid = self._try_match(track)

                if suggested_gid is not None:
                    # Start cooldown: don't assign immediately
                    proposal = self._reassign_proposals.get(lid)
                    if proposal is not None and proposal["suggested_gid"] == suggested_gid:
                        proposal["stable_count"] += 1
                    else:
                        self._reassign_proposals[lid] = {
                            "suggested_gid": suggested_gid,
                            "stable_count": 1,
                        }

                    # Check if cooldown passed
                    if self._reassign_proposals[lid]["stable_count"] >= self.MIN_STABLE_FRAMES:
                        # Assign the recovered global ID
                        self._local_to_global[lid] = suggested_gid
                        self._reassign_proposals.pop(lid, None)
                        print(f"[GLOBAL-ID] Recovered! local={lid} → global={suggested_gid}")
                    else:
                        # Cooldown not met — assign temporary global ID
                        if lid not in self._local_to_global:
                            gid = self._allocate_global_id()
                            self._local_to_global[lid] = gid
                else:
                    # No match — brand new person
                    gid = self._allocate_global_id()
                    self._local_to_global[lid] = gid
                    self._reassign_proposals.pop(lid, None)

            # ── Update gallery embedding for confirmed tracks ────────
            gid = self._local_to_global[lid]
            if track.is_confirmed and track.embedding is not None:
                if gid not in self._global_gallery:
                    self._global_gallery[gid] = {
                        "embedding": track.embedding.copy(),
                        "last_box": track.smooth_box.copy(),
                    }
                elif reid_frame:
                    # Slow EMA update — only on ReID frames
                    gallery = self._global_gallery[gid]
                    gallery["embedding"] = self._ema_update(
                        gallery["embedding"], track.embedding, alpha=0.15)
                    gallery["last_box"] = track.smooth_box.copy()

            # ── Continue checking cooldown for pending proposals ──────
            if lid in self._reassign_proposals and lid in self._local_to_global:
                proposal = self._reassign_proposals[lid]
                new_suggestion = self._try_match(track)
                if new_suggestion == proposal["suggested_gid"]:
                    proposal["stable_count"] += 1
                    if proposal["stable_count"] >= self.MIN_STABLE_FRAMES:
                        old_gid = self._local_to_global[lid]
                        new_gid = proposal["suggested_gid"]
                        if old_gid != new_gid:
                            self._local_to_global[lid] = new_gid
                            # Merge gallery: keep the new identity's embedding
                            if old_gid in self._global_gallery:
                                # Move data to new gid if it doesn't exist
                                if new_gid not in self._global_gallery:
                                    self._global_gallery[new_gid] = self._global_gallery.pop(old_gid)
                                else:
                                    del self._global_gallery[old_gid]
                            print(f"[GLOBAL-ID] Reassigned! local={lid}: "
                                  f"global {old_gid} → {new_gid} "
                                  f"(after {self.MIN_STABLE_FRAMES} stable frames)")
                        self._reassign_proposals.pop(lid, None)
                else:
                    # Suggestion changed — reset cooldown
                    if new_suggestion is not None:
                        self._reassign_proposals[lid] = {
                            "suggested_gid": new_suggestion,
                            "stable_count": 1,
                        }
                    else:
                        self._reassign_proposals.pop(lid, None)

        # ── Cleanup: remove proposals for dead local_ids ─────────────
        dead_proposals = [lid for lid in self._reassign_proposals
                          if lid not in self._active_local_ids]
        for lid in dead_proposals:
            del self._reassign_proposals[lid]

    def on_track_removed(self, local_id):
        """
        Called when a track is permanently removed by the tracker.
        The global gallery entry is KEPT so the identity can be recovered later.
        """
        self._reassign_proposals.pop(local_id, None)
        # Keep _local_to_global entry for potential future lookups
        # Keep _global_gallery entry for re-identification

    def get_active_global_ids(self):
        """Return set of global IDs currently active."""
        return {self._local_to_global[lid] for lid in self._active_local_ids
                if lid in self._local_to_global}

    # ── Internal ────────────────────────────────────────────────────

    def _allocate_global_id(self):
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    def _try_match(self, track):
        """
        Try to match track's embedding against gallery of ALL known global IDs.
        Returns the best matching global_id, or None if no match.
        Only matches against global IDs NOT currently active.
        """
        if track.embedding is None:
            return None

        emb = track.embedding
        active_gids = self.get_active_global_ids()

        best_gid = None
        best_sim = -1.0
        second_sim = -1.0

        for gid, gallery in self._global_gallery.items():
            # Don't match against currently active identities
            if gid in active_gids:
                continue

            sim = _cosine_sim(emb, gallery["embedding"])

            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_gid = gid
            elif sim > second_sim:
                second_sim = sim

        if best_sim < self.MATCH_THRESHOLD:
            return None

        # Ambiguity check
        if (best_sim - second_sim) < self.AMBIGUITY_MARGIN:
            return None

        return best_gid

    @staticmethod
    def _ema_update(old_emb, new_emb, alpha=0.15):
        """EMA update with L2 re-normalization."""
        result = (1.0 - alpha) * old_emb + alpha * new_emb
        norm = np.linalg.norm(result)
        if norm > 1e-6:
            result /= norm
        return result
