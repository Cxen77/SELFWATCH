"""
Global Multi-Camera Identity Manager — Shared Identity Space

The SINGLE authority for cross-camera identity management.
Maps local track IDs from ANY camera to ONE global identity.

Architecture:
    Each camera has its own local tracker with local track IDs.
    This manager maintains:
        1. A shared global ID counter
        2. Per-camera local-to-global ID mappings
        3. Dormant identity memory (for cross-camera re-identification)
        4. Camera observation history per global identity

Key invariant:
    same person = same global ID across ALL cameras.

Thread safety:
    All public methods are protected by a threading.Lock since
    multiple camera streams may call into this concurrently.
"""

import time
import threading
import numpy as np
from typing import Dict, Optional, List, Tuple, Set, Any
from dataclasses import dataclass, field


# ── Dormant Identity State ──────────────────────────────────────────────
@dataclass
class DormantIdentity:
    """
    A global identity that is not currently visible in any camera.
    Retained for cross-camera re-identification.
    """
    global_id: int
    embeddings: List[np.ndarray]        # Gallery of embeddings
    stable_embedding: Optional[np.ndarray]  # EWMA-smoothed embedding
    last_seen_camera: int
    last_seen_time: float
    last_seen_frame: int
    last_box: Optional[list]            # [x1, y1, x2, y2]
    last_velocity: Optional[np.ndarray] # [vx, vy]
    confidence: float                   # Decaying confidence
    total_observations: int             # Lifetime observation count
    entry_camera: int                   # First camera this identity appeared in
    trajectory_summary: Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float = 0.995):
        """Apply exponential decay to dormant confidence."""
        self.confidence *= rate

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_seen_time


# ── Cross-Camera Memory Entry ──────────────────────────────────────────
@dataclass
class GlobalMemoryEntry:
    """
    Persistent cross-camera identity memory entry.
    Stores complete history across all camera observations.
    """
    global_id: int
    embedding_history: List[np.ndarray]     # All collected embeddings (capped)
    stable_embedding: Optional[np.ndarray]  # Best representative embedding
    last_seen_camera: int
    last_seen_time: float
    last_seen_frame: int
    entry_zone: Optional[str]               # Future: zone label
    exit_zone: Optional[str]                # Future: zone label
    trajectory_summary: Dict[str, Any]
    confidence: float
    state: str                              # "active" or "dormant"
    total_observations: int
    camera_history: List[int]               # Ordered list of cameras visited
    first_seen_time: float
    last_box: Optional[list] = None
    last_velocity: Optional[np.ndarray] = None

    MAX_EMBEDDING_HISTORY = 20

    def add_embedding(self, emb: np.ndarray):
        """Add embedding to history, maintaining max size."""
        if emb is None:
            return
        self.embedding_history.append(emb.copy())
        if len(self.embedding_history) > self.MAX_EMBEDDING_HISTORY:
            self.embedding_history.pop(0)
        # Update stable embedding (EWMA)
        if self.stable_embedding is None:
            self.stable_embedding = emb.copy()
        else:
            alpha = 0.15
            self.stable_embedding = (
                (1 - alpha) * self.stable_embedding + alpha * emb
            )
            # Re-normalize
            norm = np.linalg.norm(self.stable_embedding)
            if norm > 1e-6:
                self.stable_embedding /= norm


class GlobalMultiCameraIdentityManager:
    """
    Shared global identity space across all cameras.

    Responsibilities:
        - Maintain one shared global ID registry
        - Map (camera_id, local_track_id) → global_id
        - Track dormant identities for cross-camera matching
        - Maintain persistent memory entries per global identity
        - Track camera origin for each observation
    """

    def __init__(self,
                 dormant_decay_rate: float = 0.995,
                 dormant_min_confidence: float = 0.05,
                 max_dormant: int = 200,
                 max_embedding_gallery: int = 10):
        self._lock = threading.Lock()

        # Global ID counter (monotonically increasing)
        self._next_global_id = 1

        # (camera_id, local_track_id) → global_id
        self._local_to_global: Dict[Tuple[int, int], int] = {}

        # global_id → set of (camera_id, local_track_id) currently active
        self._global_to_local: Dict[int, Set[Tuple[int, int]]] = {}

        # global_id → GlobalMemoryEntry
        self._memory: Dict[int, GlobalMemoryEntry] = {}

        # global_id → DormantIdentity
        self._dormant: Dict[int, DormantIdentity] = {}

        # Configuration
        self._dormant_decay_rate = dormant_decay_rate
        self._dormant_min_confidence = dormant_min_confidence
        self._max_dormant = max_dormant
        self._max_embedding_gallery = max_embedding_gallery

        # Active global IDs per camera
        self._active_per_camera: Dict[int, Set[int]] = {}

        # Statistics
        self._stats = {
            "total_global_ids_created": 0,
            "total_cross_camera_matches": 0,
            "total_dormant_recoveries": 0,
            "total_dormant_expired": 0,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  GLOBAL ID ALLOCATION
    # ═══════════════════════════════════════════════════════════════════

    def allocate_global_id(self) -> int:
        """Allocate a new unique global identity ID (thread-safe)."""
        with self._lock:
            gid = self._next_global_id
            self._next_global_id += 1
            self._stats["total_global_ids_created"] += 1
            return gid

    # ═══════════════════════════════════════════════════════════════════
    #  LOCAL ↔ GLOBAL MAPPING
    # ═══════════════════════════════════════════════════════════════════

    def register_local_track(self, camera_id: int, local_track_id: int,
                             global_id: Optional[int] = None,
                             embedding: Optional[np.ndarray] = None,
                             box: Optional[list] = None,
                             velocity: Optional[np.ndarray] = None) -> int:
        """
        Register a local track and map it to a global ID.

        If global_id is provided, reuse that ID (cross-camera match).
        Otherwise, allocate a new global ID.

        Returns the assigned global_id.
        """
        with self._lock:
            key = (camera_id, local_track_id)

            # Already registered?
            if key in self._local_to_global:
                return self._local_to_global[key]

            # Use provided or allocate new
            if global_id is None:
                global_id = self._next_global_id
                self._next_global_id += 1
                self._stats["total_global_ids_created"] += 1

            self._local_to_global[key] = global_id

            # Track reverse mapping
            if global_id not in self._global_to_local:
                self._global_to_local[global_id] = set()
            self._global_to_local[global_id].add(key)

            # Track active per camera
            if camera_id not in self._active_per_camera:
                self._active_per_camera[camera_id] = set()
            self._active_per_camera[camera_id].add(global_id)

            # Initialize or update memory entry
            now = time.time()
            if global_id not in self._memory:
                self._memory[global_id] = GlobalMemoryEntry(
                    global_id=global_id,
                    embedding_history=[],
                    stable_embedding=None,
                    last_seen_camera=camera_id,
                    last_seen_time=now,
                    last_seen_frame=0,
                    entry_zone=None,
                    exit_zone=None,
                    trajectory_summary={},
                    confidence=1.0,
                    state="active",
                    total_observations=1,
                    camera_history=[camera_id],
                    first_seen_time=now,
                    last_box=box,
                    last_velocity=velocity,
                )
            else:
                entry = self._memory[global_id]
                entry.state = "active"
                entry.last_seen_camera = camera_id
                entry.last_seen_time = now
                entry.confidence = 1.0
                if camera_id not in entry.camera_history:
                    entry.camera_history.append(camera_id)

            if embedding is not None:
                self._memory[global_id].add_embedding(embedding)

            # Remove from dormant if it was there
            if global_id in self._dormant:
                del self._dormant[global_id]
                self._stats["total_dormant_recoveries"] += 1

            return global_id

    def unregister_local_track(self, camera_id: int, local_track_id: int):
        """
        Unregister a local track. Called when a track is removed
        from a camera's local tracker.
        """
        with self._lock:
            key = (camera_id, local_track_id)
            global_id = self._local_to_global.pop(key, None)
            if global_id is None:
                return

            # Remove from reverse mapping
            if global_id in self._global_to_local:
                self._global_to_local[global_id].discard(key)
                if not self._global_to_local[global_id]:
                    del self._global_to_local[global_id]

            # Remove from active per camera
            if camera_id in self._active_per_camera:
                self._active_per_camera[camera_id].discard(global_id)

    def get_global_id(self, camera_id: int, local_track_id: int) -> Optional[int]:
        """Get the global ID for a local track (thread-safe)."""
        with self._lock:
            return self._local_to_global.get((camera_id, local_track_id))

    def get_local_tracks(self, global_id: int) -> Set[Tuple[int, int]]:
        """Get all (camera_id, local_track_id) pairs for a global ID."""
        with self._lock:
            return self._global_to_local.get(global_id, set()).copy()

    def is_active_anywhere(self, global_id: int) -> bool:
        """Check if a global identity is currently active in any camera."""
        with self._lock:
            return global_id in self._global_to_local and bool(
                self._global_to_local[global_id])

    # ═══════════════════════════════════════════════════════════════════
    #  MEMORY UPDATE
    # ═══════════════════════════════════════════════════════════════════

    def update_observation(self, camera_id: int, global_id: int,
                           embedding: Optional[np.ndarray] = None,
                           box: Optional[list] = None,
                           velocity: Optional[np.ndarray] = None,
                           frame_index: int = 0):
        """
        Update memory for an active global identity with new observation.
        Called each frame for every active track.
        """
        with self._lock:
            entry = self._memory.get(global_id)
            if entry is None:
                return

            entry.last_seen_camera = camera_id
            entry.last_seen_time = time.time()
            entry.last_seen_frame = frame_index
            entry.total_observations += 1
            entry.confidence = min(1.0, entry.confidence + 0.01)

            if box is not None:
                entry.last_box = box
            if velocity is not None:
                entry.last_velocity = velocity
            if embedding is not None:
                entry.add_embedding(embedding)

    # ═══════════════════════════════════════════════════════════════════
    #  DORMANT IDENTITY MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def move_to_dormant(self, global_id: int, camera_id: int,
                        frame_index: int = 0):
        """
        Move a global identity to DORMANT state.
        Called when a person exits ALL cameras.

        Dormant identities remain searchable for cross-camera matching.
        """
        with self._lock:
            entry = self._memory.get(global_id)
            if entry is None:
                return

            # Only go dormant if not active in any other camera
            active_cameras = set()
            for (cam_id, _), gid in self._local_to_global.items():
                if gid == global_id:
                    active_cameras.add(cam_id)

            if active_cameras:
                return  # Still active somewhere

            entry.state = "dormant"

            # Create dormant identity for matching
            self._dormant[global_id] = DormantIdentity(
                global_id=global_id,
                embeddings=[e.copy() for e in entry.embedding_history[-self._max_embedding_gallery:]],
                stable_embedding=entry.stable_embedding.copy() if entry.stable_embedding is not None else None,
                last_seen_camera=camera_id,
                last_seen_time=time.time(),
                last_seen_frame=frame_index,
                last_box=entry.last_box,
                last_velocity=entry.last_velocity,
                confidence=1.0,
                total_observations=entry.total_observations,
                entry_camera=entry.camera_history[0] if entry.camera_history else camera_id,
                trajectory_summary=entry.trajectory_summary.copy(),
            )

            # Enforce max dormant limit
            if len(self._dormant) > self._max_dormant:
                # Evict lowest confidence
                worst = min(self._dormant.keys(),
                            key=lambda k: self._dormant[k].confidence)
                del self._dormant[worst]
                self._stats["total_dormant_expired"] += 1

    def get_dormant_identities(self) -> Dict[int, DormantIdentity]:
        """Get all dormant identities (thread-safe copy)."""
        with self._lock:
            return dict(self._dormant)

    def reactivate_dormant(self, global_id: int):
        """Remove an identity from dormant state (being reactivated)."""
        with self._lock:
            if global_id in self._dormant:
                del self._dormant[global_id]
            entry = self._memory.get(global_id)
            if entry:
                entry.state = "active"
                entry.confidence = 1.0

    def decay_dormant(self):
        """Apply exponential decay to all dormant identities."""
        with self._lock:
            expired = []
            for gid, dormant in self._dormant.items():
                dormant.decay(self._dormant_decay_rate)
                if dormant.confidence < self._dormant_min_confidence:
                    expired.append(gid)

            for gid in expired:
                del self._dormant[gid]
                self._stats["total_dormant_expired"] += 1
                # Also update memory entry
                entry = self._memory.get(gid)
                if entry:
                    entry.state = "expired"

    # ═══════════════════════════════════════════════════════════════════
    #  QUERY API
    # ═══════════════════════════════════════════════════════════════════

    def get_memory_entry(self, global_id: int) -> Optional[GlobalMemoryEntry]:
        """Get the full memory entry for a global identity."""
        with self._lock:
            return self._memory.get(global_id)

    def get_active_global_ids(self, camera_id: Optional[int] = None) -> Set[int]:
        """Get all active global IDs, optionally filtered by camera."""
        with self._lock:
            if camera_id is not None:
                return self._active_per_camera.get(camera_id, set()).copy()
            result = set()
            for gids in self._active_per_camera.values():
                result.update(gids)
            return result

    def get_all_cameras_for_identity(self, global_id: int) -> List[int]:
        """Get the ordered camera history for a global identity."""
        with self._lock:
            entry = self._memory.get(global_id)
            return list(entry.camera_history) if entry else []

    # ═══════════════════════════════════════════════════════════════════
    #  STATISTICS & DEBUGGING
    # ═══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics snapshot."""
        with self._lock:
            return {
                **self._stats,
                "active_global_ids": sum(
                    len(s) for s in self._active_per_camera.values()),
                "dormant_count": len(self._dormant),
                "total_memory_entries": len(self._memory),
                "cameras_tracked": len(self._active_per_camera),
            }

    def get_debug_summary(self) -> str:
        """Human-readable summary for logging."""
        stats = self.get_stats()
        lines = [
            f"[GLOBAL REGISTRY] Active GIDs: {stats['active_global_ids']}",
            f"  Dormant: {stats['dormant_count']}",
            f"  Total created: {stats['total_global_ids_created']}",
            f"  Cross-camera matches: {stats['total_cross_camera_matches']}",
            f"  Dormant recoveries: {stats['total_dormant_recoveries']}",
            f"  Cameras tracked: {stats['cameras_tracked']}",
        ]
        return "\n".join(lines)

    def record_cross_camera_match(self):
        """Increment cross-camera match counter."""
        with self._lock:
            self._stats["total_cross_camera_matches"] += 1
