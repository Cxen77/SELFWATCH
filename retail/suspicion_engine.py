"""
Suspicion Engine — per-person score accumulator with decay.

Scoring:
  ITEM_REMOVED       = +20
  ENTERED_BAG        = +40
  BODY_CONCEALMENT   = +50
  ITEM_RETURNED      = -30

Levels:
  0-39   NORMAL
  40-69  SUSPICIOUS
  70+    THEFT_ALERT

Forensic logging on every score change.
"""

import time
from collections import defaultdict

from .retail_config import (
    SCORE_ITEM_REMOVED, SCORE_ITEM_RETURNED,
    SCORE_ENTERED_BAG, SCORE_BODY_CONCEALMENT, SCORE_EXIT_CONCEALED,
    SCORE_DECAY_PER_SECOND,
    THRESHOLD_SUSPICIOUS, THRESHOLD_THEFT,
)

EVENT_SCORES = {
    "ITEM_REMOVED":       SCORE_ITEM_REMOVED,
    "ITEM_RETURNED":      SCORE_ITEM_RETURNED,
    "ENTERED_BAG":        SCORE_ENTERED_BAG,
    "BODY_CONCEALMENT":   SCORE_BODY_CONCEALMENT,
    "EXIT_WITH_CONCEALED": SCORE_EXIT_CONCEALED,
}

LEVEL_NORMAL = "NORMAL"
LEVEL_SUSPICIOUS = "SUSPICIOUS"
LEVEL_THEFT = "THEFT_ALERT"


class SuspicionEngine:

    def __init__(self):
        self._scores = defaultdict(float)
        self._last_tick = time.monotonic()
        self._prev_levels = {}

    def process_events(self, events, ownership_map):
        for ev in events:
            et = ev.get("event", "")
            delta = EVENT_SCORES.get(et, 0)
            if delta == 0:
                continue

            person_gid = ev.get("person_gid")
            if person_gid is None:
                obj_gid = ev.get("object_gid")
                if obj_gid and obj_gid in ownership_map:
                    person_gid = ownership_map[obj_gid].get("owner_gid")

            if person_gid is None:
                print(f"[SUSPICION] DROPPED event={et} object_gid={ev.get('object_gid')} "
                      f"reason=no_person_attribution")
                continue

            old_score = self._scores[person_gid]
            self._scores[person_gid] = max(0.0, min(100.0,
                old_score + delta))
            new_score = self._scores[person_gid]

            old_level = self._level_for(old_score)
            new_level = self._level_for(new_score)

            print(f"[SUSPICION] gid={person_gid} event={et} "
                  f"delta={delta:+d} score={old_score:.0f}->{new_score:.0f} "
                  f"level={new_level}")

            if new_level != old_level:
                self._prev_levels[person_gid] = new_level
                print(f"[SUSPICION] gid={person_gid} LEVEL CHANGE: "
                      f"{old_level} -> {new_level}")

    def tick(self):
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        decay = SCORE_DECAY_PER_SECOND * dt
        for gid in list(self._scores.keys()):
            self._scores[gid] = max(0.0, self._scores[gid] - decay)
            if self._scores[gid] <= 0:
                del self._scores[gid]

    def _level_for(self, score):
        if score >= THRESHOLD_THEFT:
            return LEVEL_THEFT
        elif score >= THRESHOLD_SUSPICIOUS:
            return LEVEL_SUSPICIOUS
        return LEVEL_NORMAL

    def get_level(self, person_gid):
        return self._level_for(self._scores.get(person_gid, 0))

    def get_score(self, person_gid):
        return self._scores.get(person_gid, 0)

    def get_all_scores(self):
        return dict(self._scores)

    def get_alerts(self):
        alerts = []
        for gid, score in self._scores.items():
            if score >= THRESHOLD_THEFT:
                alerts.append({
                    "person_gid": gid,
                    "score": round(score, 1),
                    "level": LEVEL_THEFT,
                })
        return alerts
