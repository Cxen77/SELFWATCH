"""
Ownership Engine — person-object association via proximity, temporal consistency, and motion.

Assigns suspected_owner_gid to any non-person tracked object near a person.
Forensic logging on every ownership assignment and dissolution.
"""

import math
from collections import defaultdict

from .rfdetr_taxonomy import PERSON, ALL_TRACKED_CLASSES, class_label
from .retail_config import (
    OWNERSHIP_PROXIMITY_PX, OWNERSHIP_CONFIRM_FRAMES,
    OWNERSHIP_DISSOLVE_FRAMES, OWNERSHIP_MOTION_CORR_THRESH,
)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _mag(v):
    return math.hypot(v[0], v[1])


class OwnershipEngine:

    def __init__(self):
        self._candidates = defaultdict(lambda: defaultdict(int))
        self._owners = {}
        self._dissolve_counters = defaultdict(int)
        self._object_classes = {}

    def update(self, display_items, id_states):
        persons = {}
        objects = {}

        for gid, item in display_items.items():
            cid = item.get("class_id", 0)
            if cid == PERSON:
                persons[gid] = item
            elif cid in ALL_TRACKED_CLASSES:
                objects[gid] = item
                self._object_classes[gid] = cid

        for obj_gid, obj_item in objects.items():
            obj_center = _center(obj_item["box"])
            obj_vel = obj_item.get("vel") or [0, 0]
            cid = obj_item.get("class_id", 0)

            best_person = None
            best_score = 0.0
            best_dist = float("inf")
            n_candidates = 0

            for p_gid, p_item in persons.items():
                p_center = _center(p_item["box"])
                d = _dist(obj_center, p_center)
                if d > OWNERSHIP_PROXIMITY_PX:
                    continue

                n_candidates += 1
                proximity_score = 1.0 - (d / OWNERSHIP_PROXIMITY_PX)

                p_vel = p_item.get("vel") or [0, 0]
                mag_p = _mag(p_vel)
                mag_o = _mag(obj_vel)
                if mag_p > 0.5 and mag_o > 0.5:
                    motion_corr = _dot2(p_vel, obj_vel) / (mag_p * mag_o)
                else:
                    motion_corr = 0.5

                score = 0.6 * proximity_score + 0.4 * max(0, motion_corr)

                if score > best_score:
                    best_score = score
                    best_person = p_gid
                    best_dist = d

            if n_candidates == 0 and obj_gid not in self._owners:
                pass
            elif best_person is not None:
                prev_count = self._candidates[obj_gid][best_person]
                self._candidates[obj_gid][best_person] += 1
                self._dissolve_counters[obj_gid] = 0

                if prev_count < OWNERSHIP_CONFIRM_FRAMES <= self._candidates[obj_gid][best_person]:
                    print(f"[OWNERSHIP] object_gid={obj_gid} class={class_label(cid)} "
                          f"owner_gid={best_person} score={best_score:.2f} "
                          f"distance={best_dist:.0f}px CONFIRMED")

                if self._candidates[obj_gid][best_person] >= OWNERSHIP_CONFIRM_FRAMES:
                    self._owners[obj_gid] = {
                        "owner_gid": best_person,
                        "ownership_score": best_score,
                        "frames_held": self._candidates[obj_gid][best_person],
                        "distance": best_dist,
                    }
            else:
                if obj_gid in self._owners:
                    self._dissolve_counters[obj_gid] += 1
                    if self._dissolve_counters[obj_gid] >= OWNERSHIP_DISSOLVE_FRAMES:
                        old_owner = self._owners[obj_gid]["owner_gid"]
                        print(f"[OWNERSHIP] object_gid={obj_gid} class={class_label(cid)} "
                              f"owner_gid={old_owner} DISSOLVED "
                              f"(no proximity for {OWNERSHIP_DISSOLVE_FRAMES} frames)")
                        del self._owners[obj_gid]
                        self._candidates[obj_gid].clear()
                        self._dissolve_counters[obj_gid] = 0

        return dict(self._owners)

    def get_owner(self, object_gid):
        return self._owners.get(object_gid)

    def get_all_owned_by(self, person_gid):
        return [og for og, info in self._owners.items()
                if info["owner_gid"] == person_gid]
