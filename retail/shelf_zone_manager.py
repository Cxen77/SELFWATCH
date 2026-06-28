"""
Shelf Occupancy Tracker — zone-based object state tracking.

Tracks any non-person object in configurable shelf zones.
Detects: ON_SHELF → REMOVED → RETURNED transitions.
Forensic logging on every state change.
"""

import math
from collections import defaultdict

from .rfdetr_taxonomy import PERSON, ALL_TRACKED_CLASSES, class_label

ON_SHELF = "ON_SHELF"
REMOVED = "REMOVED"
RETURNED = "RETURNED"


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _in_zone(center, zone):
    x, y = center
    _, x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


class ShelfZoneManager:

    def __init__(self, zones=None):
        if not zones:
            zones = [("ZONE_A", 0, 0, 960, 540)]
        self._zones = zones
        self._object_states = {}
        self._object_classes = {}
        self._zone_objects = defaultdict(set)
        self._removal_frames = {}
        self._return_frames = {}
        self._object_zones = {}
        self._last_nearest_person = {}

    def update(self, display_items, ownership_map, id_states, frame_count):
        events = []

        objects = {}
        persons = {}
        for gid, item in display_items.items():
            cid = item.get("class_id", 0)
            if cid == PERSON:
                persons[gid] = item
            elif cid != PERSON and cid in ALL_TRACKED_CLASSES:
                objects[gid] = item
                self._object_classes[gid] = cid

        active_object_gids = set(objects.keys())

        for obj_gid, item in objects.items():
            center = _center(item["box"])
            cid = item.get("class_id", 0)
            owned = obj_gid in ownership_map

            best_person = None
            best_dist = float("inf")
            for p_gid, p_item in persons.items():
                d = math.hypot(center[0] - _center(p_item["box"])[0],
                               center[1] - _center(p_item["box"])[1])
                if d < best_dist:
                    best_dist = d
                    best_person = p_gid
            if best_person is not None and best_dist < 200:
                self._last_nearest_person[obj_gid] = best_person

            for zone in self._zones:
                zone_name = zone[0]
                if not _in_zone(center, zone):
                    continue

                prev_state = self._object_states.get(obj_gid)

                if not owned:
                    if prev_state == REMOVED:
                        self._object_states[obj_gid] = RETURNED
                        self._return_frames[obj_gid] = frame_count
                        self._zone_objects[zone_name].add(obj_gid)
                        self._object_zones[obj_gid] = zone_name
                        print(f"[SHELF] zone={zone_name} object_gid={obj_gid} "
                              f"class={class_label(cid)} event=RETURNED frame={frame_count}")
                        events.append({
                            "event": "ITEM_RETURNED",
                            "object_gid": obj_gid,
                            "zone": zone_name,
                            "class": class_label(cid),
                            "frame": frame_count,
                        })
                    elif prev_state is None:
                        self._object_states[obj_gid] = ON_SHELF
                        self._zone_objects[zone_name].add(obj_gid)
                        self._object_zones[obj_gid] = zone_name
                        print(f"[SHELF] zone={zone_name} object_gid={obj_gid} "
                              f"class={class_label(cid)} event=ON_SHELF frame={frame_count}")
                    elif prev_state == RETURNED:
                        self._object_states[obj_gid] = ON_SHELF
                        self._zone_objects[zone_name].add(obj_gid)
                else:
                    if prev_state in (ON_SHELF, RETURNED, None):
                        self._object_states[obj_gid] = REMOVED
                        self._removal_frames[obj_gid] = frame_count
                        for zn in self._zone_objects:
                            self._zone_objects[zn].discard(obj_gid)
                        owner_info = ownership_map.get(obj_gid, {})
                        owner_gid = owner_info.get("owner_gid")
                        owner_score = owner_info.get("ownership_score", 0)
                        print(f"[SHELF] zone={zone_name} object_gid={obj_gid} "
                              f"class={class_label(cid)} event=REMOVED "
                              f"suspected_owner_gid={owner_gid} "
                              f"ownership_score={owner_score:.2f} frame={frame_count}")
                        events.append({
                            "event": "ITEM_REMOVED",
                            "object_gid": obj_gid,
                            "zone": zone_name,
                            "class": class_label(cid),
                            "person_gid": owner_gid,
                            "frame": frame_count,
                        })
                break

        for obj_gid, state in list(self._object_states.items()):
            if state != ON_SHELF or obj_gid in active_object_gids:
                continue
            id_state_data = id_states.get(obj_gid, {})
            obj_state = id_state_data.get("state")
            if obj_state is not None and obj_state >= 1:
                self._object_states[obj_gid] = REMOVED
                self._removal_frames[obj_gid] = frame_count
                zone_name = self._object_zones.get(obj_gid, "unknown")
                cid = self._object_classes.get(obj_gid, 0)
                for zn in self._zone_objects:
                    self._zone_objects[zn].discard(obj_gid)
                person_gid = self._last_nearest_person.get(obj_gid)
                print(f"[SHELF] zone={zone_name} object_gid={obj_gid} "
                      f"class={class_label(cid)} event=REMOVED "
                      f"reason=disappeared nearest_person={person_gid} frame={frame_count}")
                events.append({
                    "event": "ITEM_REMOVED",
                    "object_gid": obj_gid,
                    "zone": zone_name,
                    "class": class_label(cid),
                    "person_gid": person_gid,
                    "frame": frame_count,
                })

        return events

    def get_zone_summary(self):
        summary = {}
        for zone in self._zones:
            name = zone[0]
            on_shelf = self._zone_objects.get(name, set())
            removed = {g for g, s in self._object_states.items() if s == REMOVED}
            summary[name] = {
                "objects_on_shelf": list(on_shelf),
                "removed_objects": list(removed),
                "total_on_shelf": len(on_shelf),
                "total_removed": len(removed),
            }
        return summary

    def is_on_shelf(self, obj_gid):
        return self._object_states.get(obj_gid) == ON_SHELF

    def get_state(self, obj_gid):
        return self._object_states.get(obj_gid)

    @property
    def zones(self):
        return self._zones
