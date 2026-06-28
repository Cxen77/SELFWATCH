"""
Alert System — THEFT_ALERT generation and ESP32 alarm triggering.

Forensic logging on every alert.
"""

import json
import time
import threading
from datetime import datetime

try:
    import urllib.request
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False


class AlertSystem:

    def __init__(self, esp32_url="http://192.168.1.100/alert", cooldown=30.0):
        self._esp32_url = esp32_url
        self._cooldown = cooldown
        self._last_alert_time = {}
        self._alerts = []

    def check_and_alert(self, suspicion_engine, ownership_map, timeline, frame_count):
        alerts = suspicion_engine.get_alerts()
        fired = []

        for alert in alerts:
            p_gid = alert["person_gid"]
            now = time.monotonic()
            last = self._last_alert_time.get(p_gid, 0)
            if now - last < self._cooldown:
                continue

            owned_objects = [og for og, info in ownership_map.items()
                            if info.get("owner_gid") == p_gid]

            bag_gids = []
            for og, info in ownership_map.items():
                if info.get("owner_gid") == p_gid:
                    pass
            for gid, item_info in ownership_map.items():
                pass

            person_events = timeline.get_events_for_person(p_gid)

            alert_record = {
                "alert": "THEFT",
                "person_gid": p_gid,
                "object_gids": owned_objects,
                "confidence": round(alert["score"] / 100.0, 2),
                "score": alert["score"],
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "frame": frame_count,
                "event_history": person_events[-10:],
            }

            self._alerts.append(alert_record)
            self._last_alert_time[p_gid] = now
            fired.append(alert_record)

            print(f"\n{'='*60}")
            print(f"[ALERT] THEFT_ALERT")
            print(f"[ALERT] person_gid={p_gid}")
            print(f"[ALERT] score={alert['score']:.0f}")
            print(f"[ALERT] object_gids={owned_objects}")
            print(f"[ALERT] frame={frame_count}")
            print(f"[ALERT] Event history:")
            for ev in person_events[-10:]:
                print(f"[ALERT]   {ev.get('timestamp', '?')} "
                      f"{ev.get('event', ev.get('event_type', '?'))}")
            print(f"{'='*60}\n")

            self._send_esp32(alert_record)

        return fired

    def _send_esp32(self, alert_record):
        if not _HAS_URLLIB:
            return

        payload = json.dumps({
            "alert": "THEFT",
            "person_gid": alert_record["person_gid"],
            "object_gid": alert_record["object_gids"][0] if alert_record["object_gids"] else None,
            "confidence": alert_record["confidence"],
        }).encode("utf-8")

        def _post():
            try:
                req = urllib.request.Request(
                    self._esp32_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
                print(f"[ESP32] Alert sent to {self._esp32_url}")
            except Exception as e:
                print(f"[ESP32] Failed to send alert: {e}")

        t = threading.Thread(target=_post, daemon=True)
        t.start()

    def get_active_alerts(self):
        return self._alerts[-10:]

    @property
    def alerts(self):
        return self._alerts
