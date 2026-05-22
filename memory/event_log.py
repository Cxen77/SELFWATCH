"""
SELFWATCH — Cognitive Event Logger

Structured JSONL event logging for all cognitive memory events.
Each line in the output file is a single JSON object with timestamp,
event type, track ID, and event-specific data.

Designed for minimal I/O impact: events are buffered and flushed
periodically, not on every write.
"""

import os
import json
import time
import threading
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class CognitiveEventLogger:
    """
    Lightweight JSONL event logger for cognitive memory events.

    Writes one JSON object per line to `logs/cognitive_events.jsonl`.
    Buffered writes (flush every `flush_interval` events or `flush_seconds`)
    to avoid I/O bottleneck during real-time tracking.

    Args:
        log_dir:        Directory for log files.
        enabled:        If False, all log() calls are no-ops.
        flush_interval: Flush buffer after this many events.
        flush_seconds:  Flush buffer after this many seconds.
    """

    def __init__(self, log_dir="logs", enabled=True,
                 flush_interval=50, flush_seconds=5.0):
        self.enabled = enabled
        self._flush_interval = flush_interval
        self._flush_seconds = flush_seconds
        self._buffer = []
        self._last_flush = time.time()
        self._file = None
        self._lock = threading.Lock()

        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"cognitive_events_{timestamp}.jsonl")
            self._file = open(path, "w", encoding="utf-8")

    def log(self, event_type, track_id=None, **data):
        """
        Log a structured event.

        Args:
            event_type: One of:
                "resurrection"       — Successful identity recovery
                "retrieval_attempt"  — Searched warm memory (success or fail)
                "retrieval_candidates" — Top-3 candidates during retrieval
                "memory_save"        — Track saved to warm memory
                "memory_decay"       — Track expired from warm memory
                "memory_prune"       — Track pruned due to capacity limit
                "memory_lock"        — Embedding update frozen
                "state_transition"   — Identity state changed
                "id_switch"          — Track ID was overwritten
                "archive"            — Entry moved to archive layer
            track_id: The track ID involved (or None for global events).
            **data:   Additional key-value data specific to the event.
        """
        if not self.enabled:
            return

        entry = {
            "t": round(time.time(), 4),
            "event": event_type,
        }
        if track_id is not None:
            entry["track_id"] = int(track_id)
        entry.update(data)

        with self._lock:
            self._buffer.append(entry)
            if (len(self._buffer) >= self._flush_interval or
                    time.time() - self._last_flush >= self._flush_seconds):
                self._flush()

    def _flush(self):
        """Write buffered events to disk."""
        if self._file is None or not self._buffer:
            return
        for entry in self._buffer:
            self._file.write(json.dumps(entry, cls=NumpyEncoder, separators=(",", ":")) + "\n")
        self._file.flush()
        self._buffer.clear()
        self._last_flush = time.time()

    def close(self):
        """Flush remaining events and close file handle."""
        if not self.enabled:
            return
        with self._lock:
            self._flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
