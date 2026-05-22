"""
Layer 3: Active Memory — Fingerprint-Backed Identity Storage

Stores rich, short-term data for currently tracked global identities.
Each identity now has a full IdentityFingerprint providing multi-signal
matching (embedding EWMA, color histograms, body features, motion).

Architecture rule: advisory only — never mutates tracker state.
"""

import numpy as np
from memory.identity_fingerprint import IdentityFingerprint


def _l2_normalize(v):
    v = v.copy().astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 1e-6:
        v /= norm
    return v


class ActiveMemory:
    """
    Maintains detailed short-term state for active global IDs,
    backed by IdentityFingerprint for multi-signal matching.
    """

    def __init__(self):
        self.identities = {}  # gid -> identity data

    def update(self, global_id, embedding, box, score, frame_delta=1,
               crop=None, velocity=None):
        """
        Update the active memory for a global ID.

        Args:
            global_id: Global identity ID
            embedding: L2-normalized 512-d embedding (or None)
            box: [x1, y1, x2, y2]
            score: detection confidence
            frame_delta: frames since last update
            crop: BGR person crop image (for color histograms)
            velocity: [vx, vy] velocity vector
        """
        emb = _l2_normalize(embedding) if embedding is not None else None

        if global_id not in self.identities:
            fp = IdentityFingerprint()
            fp.update(emb, box, crop=crop, score=score, velocity=velocity)

            self.identities[global_id] = {
                "fingerprint": fp,
                # Legacy fields (for warm memory compatibility)
                "stable_embedding": emb.copy() if emb is not None else None,
                "recent_embedding": emb.copy() if emb is not None else None,
                "best_embedding": emb.copy() if emb is not None else None,
                "best_score": score,
                "gallery": [emb.copy()] if emb is not None else [],
                "trajectory": [np.array(box, dtype=np.float32)],
                "confidence_history": [score],
                "age": 1,
                "importance": score,
                "last_velocity": np.array(velocity or [0, 0], dtype=np.float32),
            }
            return

        identity = self.identities[global_id]
        identity["age"] += frame_delta

        # Update fingerprint
        identity["fingerprint"].update(
            emb, box, crop=crop, score=score, velocity=velocity)

        # Update trajectory (keep last 30 frames)
        identity["trajectory"].append(np.array(box, dtype=np.float32))
        if len(identity["trajectory"]) > 30:
            identity["trajectory"].pop(0)

        identity["confidence_history"].append(score)
        if len(identity["confidence_history"]) > 30:
            identity["confidence_history"].pop(0)

        # Update velocity (EMA)
        if len(identity["trajectory"]) >= 2:
            last = identity["trajectory"][-1]
            prev = identity["trajectory"][-2]
            cx1 = (last[0] + last[2]) / 2
            cy1 = (last[1] + last[3]) / 2
            cx2 = (prev[0] + prev[2]) / 2
            cy2 = (prev[1] + prev[3]) / 2
            vx = (cx1 - cx2) / frame_delta
            vy = (cy1 - cy2) / frame_delta
            identity["last_velocity"] = (
                0.85 * identity["last_velocity"]
                + 0.15 * np.array([vx, vy], dtype=np.float32)
            )

        # Update legacy embedding fields (synced from fingerprint)
        fp = identity["fingerprint"]
        if fp.ewma_embedding is not None:
            identity["stable_embedding"] = fp.ewma_embedding.copy()
        if fp.recent_embedding is not None:
            identity["recent_embedding"] = fp.recent_embedding.copy()
        if fp.best_embedding is not None:
            identity["best_embedding"] = fp.best_embedding.copy()
            identity["best_score"] = fp.best_score
        identity["gallery"] = [g.copy() for g in fp.gallery]

        # Update importance
        avg_conf = float(np.mean(identity["confidence_history"]))
        identity["importance"] = (identity["age"] / 30.0) + avg_conf

    def get_identity(self, global_id):
        return self.identities.get(global_id)

    def get_fingerprint(self, global_id):
        """Get the IdentityFingerprint for a global ID."""
        identity = self.identities.get(global_id)
        if identity:
            return identity.get("fingerprint")
        return None

    def remove(self, global_id):
        return self.identities.pop(global_id, None)

    def get_all_active_ids(self):
        return set(self.identities.keys())
