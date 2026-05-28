"""
Multi-Camera Event System — Entry/Exit Tracking

Tracks when identities enter and leave camera views.
Generates timestamped events for cross-camera transitions.

Events are stored in a thread-safe bus and can be consumed
by the cross-camera ReID matcher and future topology layers.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class EventType(Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    MATCH = "MATCH"           # Cross-camera match found
    NEW_GLOBAL = "NEW_GLOBAL" # New global identity created
    DORMANT = "DORMANT"       # Identity moved to dormant state
    REACTIVATE = "REACTIVATE" # Dormant identity reactivated


@dataclass
class CameraEvent:
    """A single cross-camera event."""
    event_type: EventType
    global_id: int
    camera_id: int
    timestamp: float
    frame_index: int
    local_track_id: Optional[int] = None
    embedding: Optional[Any] = None   # numpy array, kept optional for serialization
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        meta_str = ""
        if self.metadata:
            meta_str = " " + " ".join(f"{k}={v}" for k, v in self.metadata.items())
        return (
            f"[{self.event_type.value}] GID {self.global_id} "
            f"Camera{self.camera_id} at t={self.timestamp:.2f} "
            f"frame={self.frame_index}"
            f"{meta_str}"
        )


class CameraEventBus:
    """
    Thread-safe event bus for cross-camera identity events.

    All camera streams publish events here. The cross-camera
    ReID matcher and logging system consume them.
    """

    def __init__(self, max_history: int = 5000):
        self._events: List[CameraEvent] = []
        self._lock = threading.Lock()
        self._max_history = max_history

        # Callbacks registered for real-time event processing
        self._listeners: List[callable] = []

    def publish(self, event: CameraEvent):
        """Publish an event to the bus (thread-safe)."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_history:
                self._events.pop(0)

        # Notify listeners (outside lock to avoid deadlock)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"[EVENT BUS] Listener error: {e}")

    def register_listener(self, callback: callable):
        """Register a callback for real-time event processing."""
        self._listeners.append(callback)

    def get_recent_exits(self, max_age_seconds: float = 60.0,
                         camera_id: Optional[int] = None) -> List[CameraEvent]:
        """Get recent EXIT events, optionally filtered by camera."""
        now = time.time()
        with self._lock:
            results = []
            for e in reversed(self._events):
                if now - e.timestamp > max_age_seconds:
                    break
                if e.event_type == EventType.EXIT:
                    if camera_id is None or e.camera_id == camera_id:
                        results.append(e)
            return results

    def get_recent_entries(self, max_age_seconds: float = 60.0,
                          camera_id: Optional[int] = None) -> List[CameraEvent]:
        """Get recent ENTER events, optionally filtered by camera."""
        now = time.time()
        with self._lock:
            results = []
            for e in reversed(self._events):
                if now - e.timestamp > max_age_seconds:
                    break
                if e.event_type == EventType.ENTER:
                    if camera_id is None or e.camera_id == camera_id:
                        results.append(e)
            return results

    def get_events_for_global_id(self, global_id: int,
                                 limit: int = 50) -> List[CameraEvent]:
        """Get all events for a specific global identity."""
        with self._lock:
            results = [e for e in self._events if e.global_id == global_id]
            return results[-limit:]

    def get_transition_pairs(self, max_gap_seconds: float = 120.0) -> List[tuple]:
        """
        Find EXIT→ENTER pairs that suggest cross-camera transitions.

        Returns list of (exit_event, enter_event) tuples where:
        - Same global_id
        - Different cameras
        - EXIT happened before ENTER
        - Time gap is within max_gap_seconds
        """
        with self._lock:
            exits_by_gid = {}
            pairs = []

            for e in self._events:
                if e.event_type == EventType.EXIT:
                    if e.global_id not in exits_by_gid:
                        exits_by_gid[e.global_id] = []
                    exits_by_gid[e.global_id].append(e)

                elif e.event_type == EventType.ENTER:
                    gid = e.global_id
                    if gid in exits_by_gid:
                        for exit_e in exits_by_gid[gid]:
                            if (exit_e.camera_id != e.camera_id
                                    and exit_e.timestamp < e.timestamp
                                    and e.timestamp - exit_e.timestamp <= max_gap_seconds):
                                pairs.append((exit_e, e))

            return pairs

    @property
    def total_events(self) -> int:
        with self._lock:
            return len(self._events)

    def get_summary(self) -> Dict[str, int]:
        """Get event counts by type."""
        with self._lock:
            counts = {}
            for e in self._events:
                key = e.event_type.value
                counts[key] = counts.get(key, 0) + 1
            return counts
