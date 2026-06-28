"""
Concealment Detector — bag and body concealment via disappearance inference.

Bag concealment:  object near bag → object disappears → bag stays visible → ENTERED_BAG
Body concealment: object near torso → object disappears → not on shelf → BODY_CONCEALMENT

No pose estimation. Uses bounding box geometry only.
Forensic logging on every concealment event.
"""

import math
from collections import defaultdict

from .rfdetr_taxonomy import PERSON, BAG_CLASSES, ALL_TRACKED_CLASSES, class_label
from .retail_config import (
    CONCEAL_BAG_PROXIMITY_PX, CONCEAL_BODY_PROXIMITY_PX,
    CONCEAL_BODY_TORSO_RATIO, CONCEAL_CONFIRM_FRAMES,
)

STATE_ACTIVE = 0
STATE_THINKING = 1
STATE_PHANTOM = 2


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ConcealmentDetector:

    def __init__(self):
        self._last_seen_near = {}
        self._conceal_counters = defaultdict(int)
        self._fired_events = set()

    def update(self, display_items, id_states, shelf_manager, ownership_map, frame_count):
        events = []

        persons = {}
        bags = {}
        non_person_objects = {}
        active_object_gids = set()

        for gid, item in display_items.items():
            cid = item.get("class_id", 0)
            if cid == PERSON:
                persons[gid] = item
            elif cid in BAG_CLASSES:
                bags[gid] = item
                active_object_gids.add(gid)
            elif cid in ALL_TRACKED_CLASSES:
                non_person_objects[gid] = item
                active_object_gids.add(gid)

        for obj_gid, item in non_person_objects.items():
            center = _center(item["box"])
            cid = item.get("class_id", 0)

            for bag_gid, bag_item in bags.items():
                bag_center = _center(bag_item["box"])
                d = _dist(center, bag_center)
                if d < CONCEAL_BAG_PROXIMITY_PX:
                    owner_info = ownership_map.get(bag_gid, {})
                    self._last_seen_near[obj_gid] = {
                        "type": "bag",
                        "bag_gid": bag_gid,
                        "owner_gid": owner_info.get("owner_gid"),
                        "distance": d,
                        "frame": frame_count,
                        "class_id": cid,
                    }

            for p_gid, p_item in persons.items():
                p_box = p_item["box"]
                p_h = p_box[3] - p_box[1]
                torso_top = p_box[1] + p_h * (1.0 - CONCEAL_BODY_TORSO_RATIO)
                torso_center = (
                    (p_box[0] + p_box[2]) / 2,
                    (torso_top + p_box[3]) / 2,
                )
                d = _dist(center, torso_center)
                if d < CONCEAL_BODY_PROXIMITY_PX:
                    prev = self._last_seen_near.get(obj_gid)
                    if prev is None or prev["type"] != "bag":
                        self._last_seen_near[obj_gid] = {
                            "type": "person",
                            "person_gid": p_gid,
                            "distance": d,
                            "frame": frame_count,
                            "class_id": cid,
                        }

        for gid, state_data in id_states.items():
            st = state_data.get("state")
            if st not in (STATE_THINKING, STATE_PHANTOM):
                continue
            if gid not in self._last_seen_near:
                continue
            if gid in active_object_gids:
                continue
            if shelf_manager.is_on_shelf(gid):
                continue

            near_info = self._last_seen_near[gid]
            self._conceal_counters[gid] += 1

            if self._conceal_counters[gid] < CONCEAL_CONFIRM_FRAMES:
                continue

            event_key = (gid, near_info["type"], near_info.get("bag_gid"), near_info.get("person_gid"))
            if event_key in self._fired_events:
                continue

            frames_since = frame_count - near_info["frame"]
            confidence = max(0.3, min(0.95,
                1.0 - (near_info["distance"] / 200.0) - (frames_since / 100.0)))
            cid = near_info.get("class_id", 0)

            if near_info["type"] == "bag":
                bag_gid = near_info["bag_gid"]
                if bag_gid not in display_items:
                    continue
                owner_gid = near_info.get("owner_gid")
                if owner_gid is None:
                    bag_owner = ownership_map.get(bag_gid, {})
                    owner_gid = bag_owner.get("owner_gid")

                print(f"[CONCEALMENT] object_gid={gid} class={class_label(cid)} "
                      f"bag_gid={bag_gid} owner_gid={owner_gid} "
                      f"event=ENTERED_BAG confidence={confidence:.2f} frame={frame_count}")
                events.append({
                    "event": "ENTERED_BAG",
                    "object_gid": gid,
                    "bag_gid": bag_gid,
                    "person_gid": owner_gid,
                    "confidence": round(confidence, 2),
                    "frame": frame_count,
                })
                self._fired_events.add(event_key)

            elif near_info["type"] == "person":
                p_gid = near_info["person_gid"]
                if p_gid not in display_items:
                    continue

                print(f"[CONCEALMENT] object_gid={gid} class={class_label(cid)} "
                      f"owner_gid={p_gid} event=BODY_CONCEALMENT "
                      f"confidence={confidence:.2f} frame={frame_count}")
                events.append({
                    "event": "BODY_CONCEALMENT",
                    "object_gid": gid,
                    "person_gid": p_gid,
                    "confidence": round(confidence, 2),
                    "frame": frame_count,
                })
                self._fired_events.add(event_key)

        for obj_gid in list(self._conceal_counters.keys()):
            if obj_gid in active_object_gids:
                self._conceal_counters.pop(obj_gid, None)
                for key in list(self._fired_events):
                    if key[0] == obj_gid:
                        self._fired_events.discard(key)

        return events
