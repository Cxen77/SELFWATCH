"""
Evidence Timeline — append-only event log for retail theft detection.
"""

import json
import time
from datetime import datetime


class EvidenceTimeline:

    def __init__(self):
        self._events = []

    def record(self, frame, event_type, **details):
        self._events.append({
            "frame": frame,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": event_type,
            **details,
        })

    def get_events_for_person(self, person_gid):
        return [e for e in self._events if e.get("person_gid") == person_gid]

    def get_recent(self, n=20):
        return self._events[-n:]

    @property
    def events(self):
        return self._events

    def export_json(self, path="logs/evidence_timeline.json"):
        with open(path, "w") as f:
            json.dump(self._events, f, indent=2)
        print(f"[RETAIL] Evidence timeline exported to {path} ({len(self._events)} events)")
