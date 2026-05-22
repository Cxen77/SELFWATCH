"""
Layer 4: Warm Memory

Stores compressed, long-term data for lost global identities.
Handles exponential decay and archiving of forgotten identities.
"""

import time
import math
import numpy as np

class WarmMemory:
    """
    Maintains long-term state for identities that are currently out of view.
    """
    DECAY_INTERVAL = 30  # Only run decay every N calls

    def __init__(self, max_size=100, base_decay_rate=0.05, archive_threshold=0.3):
        self.identities = {}
        self.archive = []
        self.max_size = max_size
        self.max_archive = 500
        self.base_decay_rate = base_decay_rate
        self.archive_threshold = archive_threshold
        self.difficulty_multiplier = 1.0
        self._decay_counter = 0

    def set_difficulty(self, multiplier):
        """Adjust decay rate based on scene difficulty (crowd size, confidence)."""
        self.difficulty_multiplier = max(1.0, multiplier)

    def save_identity(self, global_id, active_data, current_time):
        """
        Transition an identity from ActiveMemory to WarmMemory.
        """
        # Enforce max size
        if len(self.identities) >= self.max_size:
            # Prune lowest confidence
            worst_id = min(self.identities.keys(), key=lambda k: self.identities[k]["decayed_confidence"])
            self._archive(worst_id)

        importance = active_data.get("importance", 1.0)
        
        # Compress embeddings
        embeddings = {
            "stable": active_data["stable_embedding"],
            "recent": active_data["recent_embedding"],
            "best": active_data["best_embedding"],
            "gallery": active_data["gallery"],
        }
        
        # Compress trajectory -> just last known pos & vel
        last_box = active_data["trajectory"][-1] if active_data["trajectory"] else None
        velocity = active_data["last_velocity"]

        self.identities[global_id] = {
            "embeddings": embeddings,
            "last_box": last_box,
            "velocity": velocity,
            "importance": importance,
            "decayed_confidence": 1.0,
            "lost_time": current_time,
            "last_decay_time": current_time,
        }
        # Log saved (no print on hot path)

    def decay(self, current_time):
        """
        Apply exponential decay to all warm memory entries.
        Throttled to run every DECAY_INTERVAL calls.
        """
        self._decay_counter += 1
        if self._decay_counter % self.DECAY_INTERVAL != 0:
            return

        adjusted_decay = self.base_decay_rate / self.difficulty_multiplier
        expired_ids = []

        for gid, data in self.identities.items():
            dt = current_time - data["last_decay_time"]
            data["last_decay_time"] = current_time

            effective_decay = adjusted_decay / (1.0 + data["importance"])
            data["decayed_confidence"] *= math.exp(-effective_decay * dt)

            if data["decayed_confidence"] < self.archive_threshold:
                expired_ids.append(gid)

        for gid in expired_ids:
            self._archive(gid)

    def get_identity(self, global_id):
        return self.identities.get(global_id)
        
    def get_all(self):
        return self.identities

    def resurrect(self, global_id):
        """
        Remove an identity from warm memory because it was retrieved.
        """
        return self.identities.pop(global_id, None)

    def _archive(self, global_id):
        """Move forgotten identity to cold archive."""
        data = self.identities.pop(global_id, None)
        if data:
            self.archive.append({
                "global_id": global_id,
                "importance": data["importance"],
                "archived_at": time.time()
            })
            if len(self.archive) > self.max_archive:
                self.archive.pop(0)

    @property
    def count(self):
        return len(self.identities)
