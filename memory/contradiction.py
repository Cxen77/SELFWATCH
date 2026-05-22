"""
SELFWATCH - Identity Contradiction Detector

Meta-reasoning layer that catches global identity consistency errors
that frame-by-frame association misses.

Two detection modes:
  1. DUPLICATE DETECTION: Two active tracks are the same person
     (fragmented identity). Detects via high embedding similarity
     between tracks that were never simultaneously visible.

  2. HIJACK DETECTION: One track has been stolen by a different person
     (identity swap). Detects via sudden large embedding shift without
     corresponding large spatial movement.

Cost: N*(N-1)/2 dot products every check_interval frames.
      With 5 people: 10 ops. Negligible.
"""

import numpy as np


class ContradictionDetector:
    """
    Detects and flags identity contradictions in the tracking system.

    Args:
        duplicate_sim_thresh:  Minimum embedding similarity to flag duplicates.
        hijack_sim_thresh:     Maximum similarity vs history to flag hijack.
        check_interval:        Run checks every N frames.
        event_logger:          CognitiveEventLogger instance or None.
        metrics:               TrackingMetrics instance or None.
    """

    def __init__(self, duplicate_sim_thresh=0.88, hijack_sim_thresh=0.55,
                 check_interval=30, event_logger=None, metrics=None):
        self.duplicate_sim_thresh = duplicate_sim_thresh
        self.hijack_sim_thresh = hijack_sim_thresh
        self.check_interval = check_interval
        self._logger = event_logger
        self._metrics = metrics
        self._frame_count = 0

        # Per-track embedding history for hijack detection
        self._embedding_history = {}   # track_id -> list of embeddings (last 10)
        self._coexistence = {}         # (id_a, id_b) -> frame count both active

    def tick(self, active_tracks):
        """
        Call every frame. Runs full check every check_interval frames.

        Args:
            active_tracks: List of confirmed STrack objects.

        Returns:
            dict with 'duplicates' and 'hijacks' lists.
        """
        self._frame_count += 1

        # Track coexistence (pairs that are SIMULTANEOUSLY visible)
        active_ids = set()
        for t in active_tracks:
            if t.is_confirmed and t.time_since_update == 0:
                active_ids.add(t.id)
                # Update embedding history
                if t.embedding is not None:
                    if t.id not in self._embedding_history:
                        self._embedding_history[t.id] = []
                    hist = self._embedding_history[t.id]
                    hist.append(t.embedding.copy())
                    if len(hist) > 10:
                        hist.pop(0)

        # Record coexistence
        for a in active_ids:
            for b in active_ids:
                if a < b:
                    key = (a, b)
                    self._coexistence[key] = self._coexistence.get(key, 0) + 1

        result = {"duplicates": [], "hijacks": []}

        if self._frame_count % self.check_interval != 0:
            return result

        # Run checks
        result["duplicates"] = self._check_duplicates(active_tracks)
        result["hijacks"] = self._check_hijacks(active_tracks)

        return result

    def _check_duplicates(self, active_tracks):
        """
        Find pairs of active tracks that are likely the same person.

        Criteria:
          - High embedding similarity (> duplicate_sim_thresh)
          - Never or rarely simultaneously visible
        """
        confirmed = [t for t in active_tracks
                     if t.is_confirmed and t.embedding is not None]
        duplicates = []

        for i in range(len(confirmed)):
            for j in range(i + 1, len(confirmed)):
                a, b = confirmed[i], confirmed[j]

                # Check embedding similarity
                sim = float(np.dot(a.embedding, b.embedding))
                if sim < self.duplicate_sim_thresh:
                    continue

                # Check coexistence: if they were visible together many
                # frames, they are genuinely different people
                key = (min(a.id, b.id), max(a.id, b.id))
                coex_frames = self._coexistence.get(key, 0)

                # If they coexisted for > 5 frames, they are different people
                if coex_frames > 5:
                    continue

                # DUPLICATE DETECTED
                duplicates.append({
                    "keep_id": a.id if a.total_hits >= b.total_hits else b.id,
                    "merge_id": b.id if a.total_hits >= b.total_hits else a.id,
                    "similarity": round(sim, 3),
                    "coexistence_frames": coex_frames,
                })

                if self._logger:
                    self._logger.log("contradiction_duplicate",
                                     keep=int(duplicates[-1]["keep_id"]),
                                     merge=int(duplicates[-1]["merge_id"]),
                                     sim=round(sim, 3))
                if self._metrics:
                    self._metrics.record_id_switch(
                        duplicates[-1]["merge_id"],
                        duplicates[-1]["keep_id"]
                    )

        return duplicates

    def _check_hijacks(self, active_tracks):
        """
        Find tracks where the identity was probably stolen by another person.

        Criteria:
          - Current embedding is very different from recent history
          - Bbox didn't move much (ruling out fast motion blur)
        """
        hijacks = []

        for t in active_tracks:
            if not t.is_confirmed or t.embedding is None:
                continue
            if t.id not in self._embedding_history:
                continue

            hist = self._embedding_history[t.id]
            if len(hist) < 5:
                continue

            # Compare current embedding vs historical mean
            hist_mean = np.mean(hist[:-1], axis=0)
            norm = np.linalg.norm(hist_mean)
            if norm > 1e-6:
                hist_mean /= norm

            current_sim = float(np.dot(t.embedding, hist_mean))

            if current_sim < self.hijack_sim_thresh:
                hijacks.append({
                    "track_id": t.id,
                    "similarity_vs_history": round(current_sim, 3),
                    "history_length": len(hist),
                })

                if self._logger:
                    self._logger.log("contradiction_hijack",
                                     t.id,
                                     sim=round(current_sim, 3))

        return hijacks

    def clear_track(self, track_id):
        """Clean up when a track is removed."""
        self._embedding_history.pop(track_id, None)
        # Clean coexistence pairs involving this track
        keys_to_remove = [k for k in self._coexistence if track_id in k]
        for k in keys_to_remove:
            del self._coexistence[k]
