"""
Cross-Camera ReID Matcher — Baseline Cosine Similarity Matching

Phase 1 baseline implementation for matching identities across cameras.

Algorithm:
    1. Person disappears from Camera A → moves to dormant
    2. New track appears in Camera B
    3. Compare embedding against ALL dormant identities
    4. If cosine similarity exceeds threshold → reuse global ID
    5. Apply temporal gating (don't match if time gap is implausible)

NO advanced topology logic. NO spatial transition models.
Just embedding similarity + temporal plausibility.
"""

import time
import numpy as np
import threading
from typing import Optional, Tuple, Dict, List, Any

from .global_registry import GlobalMultiCameraIdentityManager, DormantIdentity
from .events import CameraEventBus, CameraEvent, EventType


class CrossCameraReIDMatcher:
    """
    Baseline cross-camera re-identification using cosine similarity.

    Matches new tracks against dormant identities from other cameras
    using embedding distance, temporal gating, and confidence thresholds.
    """

    def __init__(self,
                 global_registry: GlobalMultiCameraIdentityManager,
                 event_bus: CameraEventBus,
                 similarity_threshold: float = 0.70,
                 max_time_gap: float = 300.0,
                 min_dormant_confidence: float = 0.10,
                 same_camera_enabled: bool = True):
        """
        Args:
            global_registry: Shared global identity manager
            event_bus: Event bus for publishing match events
            similarity_threshold: Min cosine similarity for match (0-1)
            max_time_gap: Max seconds between exit and re-entry
            min_dormant_confidence: Min dormant confidence to attempt match
            same_camera_enabled: Allow matching within same camera
        """
        self._registry = global_registry
        self._event_bus = event_bus
        self._similarity_threshold = similarity_threshold
        self._max_time_gap = max_time_gap
        self._min_dormant_confidence = min_dormant_confidence
        self._same_camera_enabled = same_camera_enabled
        self._lock = threading.Lock()

        # Match logging
        self._match_log: List[Dict[str, Any]] = []
        self._max_log = 500

    def attempt_match(self, camera_id: int, local_track_id: int,
                      embedding: np.ndarray,
                      frame_index: int = 0) -> Optional[int]:
        """
        Attempt to match a new track against dormant identities.

        Args:
            camera_id: Camera where the new track appeared
            local_track_id: Local track ID in that camera
            embedding: L2-normalized embedding of the new track
            frame_index: Current frame index

        Returns:
            Matched global_id if found, None otherwise
        """
        if embedding is None:
            return None

        now = time.time()
        dormant = self._registry.get_dormant_identities()

        if not dormant:
            return None

        best_gid = None
        best_sim = -1.0
        best_dormant = None

        for gid, dorm in dormant.items():
            # ── Temporal gating ─────────────────────────────────────────
            time_gap = now - dorm.last_seen_time
            if time_gap > self._max_time_gap:
                continue

            # ── Confidence gating ───────────────────────────────────────
            if dorm.confidence < self._min_dormant_confidence:
                continue

            # ── Same-camera gating ──────────────────────────────────────
            if not self._same_camera_enabled:
                if dorm.last_seen_camera == camera_id:
                    continue

            # ── Skip if already active in requesting camera ─────────────
            if self._registry.is_active_anywhere(gid):
                continue

            # ── Embedding comparison ────────────────────────────────────
            similarity = self._compute_best_similarity(embedding, dorm)

            if similarity > best_sim:
                best_sim = similarity
                best_gid = gid
                best_dormant = dorm

        # ── Decision ────────────────────────────────────────────────────
        if best_gid is not None and best_sim >= self._similarity_threshold:
            # Match found!
            self._log_match(
                camera_id, local_track_id, best_gid,
                best_sim, best_dormant, "ACCEPT", frame_index)

            # Publish match event
            self._event_bus.publish(CameraEvent(
                event_type=EventType.MATCH,
                global_id=best_gid,
                camera_id=camera_id,
                timestamp=now,
                frame_index=frame_index,
                local_track_id=local_track_id,
                metadata={
                    "similarity": round(best_sim, 4),
                    "source_camera": best_dormant.last_seen_camera,
                    "time_gap": round(now - best_dormant.last_seen_time, 2),
                },
            ))

            # Record cross-camera match statistic
            if best_dormant.last_seen_camera != camera_id:
                self._registry.record_cross_camera_match()

            print(f"[CROSS-CAM MATCH] GID {best_gid}: "
                  f"Camera{best_dormant.last_seen_camera} -> Camera{camera_id} "
                  f"sim={best_sim:.3f} gap={now - best_dormant.last_seen_time:.1f}s")

            return best_gid

        elif best_gid is not None:
            # Best match found but below threshold
            self._log_match(
                camera_id, local_track_id, best_gid,
                best_sim, best_dormant,
                f"REJECT_LOW_SIM({best_sim:.3f}<{self._similarity_threshold})",
                frame_index)

        return None

    def _compute_best_similarity(self, query_embedding: np.ndarray,
                                 dormant: DormantIdentity) -> float:
        """
        Compute best cosine similarity between query and dormant identity.

        Uses both the stable (EWMA) embedding and the gallery for robustness.
        """
        best_sim = -1.0

        # Compare against stable embedding
        if dormant.stable_embedding is not None:
            sim = float(np.dot(query_embedding, dormant.stable_embedding))
            best_sim = max(best_sim, sim)

        # Compare against gallery embeddings
        for gallery_emb in dormant.embeddings:
            sim = float(np.dot(query_embedding, gallery_emb))
            best_sim = max(best_sim, sim)

        return best_sim

    def _log_match(self, camera_id, local_track_id, proposed_gid,
                   similarity, dormant, decision, frame_index):
        """Log match attempt for debugging."""
        entry = {
            "timestamp": time.time(),
            "camera_id": camera_id,
            "local_track_id": local_track_id,
            "proposed_gid": proposed_gid,
            "similarity": round(similarity, 4),
            "source_camera": dormant.last_seen_camera if dormant else None,
            "dormant_confidence": round(dormant.confidence, 4) if dormant else None,
            "time_gap": round(time.time() - dormant.last_seen_time, 2) if dormant else None,
            "decision": decision,
            "frame_index": frame_index,
        }

        with self._lock:
            self._match_log.append(entry)
            if len(self._match_log) > self._max_log:
                self._match_log.pop(0)

        # Print rejections for debugging
        if "REJECT" in decision:
            import numpy as np
            print(f"[INSTRUMENTATION] CROSS-CAM NEW GID SPAWNED:")
            print(f"  - local_track_id: {local_track_id}")
            print(f"  - proposed_gid (from dormant): {proposed_gid}")
            print(f"  - cosine_similarity: {similarity:.3f}")
            print(f"  - reason: {decision}")
        elif "ACCEPT" in decision:
            print(f"[INSTRUMENTATION] CROSS-CAM GID REUSED:")
            print(f"  - local_track_id: {local_track_id}")
            print(f"  - reused_gid: {proposed_gid}")
            print(f"  - cosine_similarity: {similarity:.3f}")

    def get_match_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent match log entries."""
        with self._lock:
            return list(self._match_log[-limit:])

    def get_stats(self) -> Dict[str, Any]:
        """Get matcher statistics."""
        with self._lock:
            accepts = sum(1 for e in self._match_log if "ACCEPT" in e["decision"])
            rejects = sum(1 for e in self._match_log if "REJECT" in e["decision"])
            return {
                "total_attempts": len(self._match_log),
                "accepts": accepts,
                "rejects": rejects,
                "acceptance_rate": accepts / max(1, accepts + rejects),
            }
