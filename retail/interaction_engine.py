"""
Interaction Engine — Detects person-shelf spatial interactions.

Uses geometric reach zone estimation (no pose estimator required).
The "reach zone" is the upper-outer quadrant of a person bounding box,
approximating the arm/hand region during shelf interaction.

When a person's reach zone intersects a shelf zone AND their lateral
velocity is low (they are stopped/browsing, not walking past), an
interaction event is emitted.

Forensic logging: [INTERACT] on every state change.
"""

import math
from collections import defaultdict


# Minimum lateral velocity to count as "stopped at shelf" (px/frame)
_MAX_WALK_VELOCITY = 4.0

# How many frames of overlap before we consider it a real interaction
_CONFIRM_FRAMES = 3

# Padding to expand reach zone beyond person bbox (fraction of width)
_REACH_PAD_X = 0.15
# Reach zone spans from 15% to 55% of person height (shoulder to waist)
_REACH_TOP_FRAC = 0.15
_REACH_BOT_FRAC = 0.55


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_intersects(a, b):
    """Check if two boxes (x1,y1,x2,y2) intersect."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _reach_zone(person_box):
    """
    Compute the biomechanical reach zone from a person bounding box.
    Returns (x1, y1, x2, y2) of the region where hands are likely active.
    """
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1
    rz_x1 = x1 - w * _REACH_PAD_X
    rz_x2 = x2 + w * _REACH_PAD_X
    rz_y1 = y1 + h * _REACH_TOP_FRAC
    rz_y2 = y1 + h * _REACH_BOT_FRAC
    return (rz_x1, rz_y1, rz_x2, rz_y2)


class InteractionEvent:
    """A single person-shelf interaction."""
    __slots__ = [
        "person_gid", "zone_name", "zone_box",
        "reach_box", "person_box", "start_frame",
        "active", "confirmed",
    ]

    def __init__(self, person_gid, zone_name, zone_box,
                 reach_box, person_box, frame):
        self.person_gid = person_gid
        self.zone_name = zone_name
        self.zone_box = zone_box
        self.reach_box = reach_box
        self.person_box = person_box
        self.start_frame = frame
        self.active = True
        self.confirmed = False


class InteractionEngine:
    """
    Detects when a person is interacting with a shelf zone.

    Usage:
        events = engine.update(display_items, shelf_zones, frame_count)
        # events is a list of InteractionEvent objects that are currently active
    """

    def __init__(self, confirm_frames=_CONFIRM_FRAMES):
        self._confirm_frames = confirm_frames
        # (person_gid, zone_name) -> frame count of continuous overlap
        self._overlap_counters = defaultdict(int)
        # (person_gid, zone_name) -> InteractionEvent
        self._active_interactions = {}
        # Set of (person_gid, zone_name) pairs that had interaction last frame
        self._prev_active = set()

    def update(self, display_items, shelf_zones, frame_count):
        """
        Args:
            display_items: dict of gid -> {"box", "vel", "class_id", ...}
            shelf_zones: list of (zone_name, x1, y1, x2, y2)
            frame_count: current frame number

        Returns:
            list of active InteractionEvent objects
        """
        from .rfdetr_taxonomy import PERSON

        # Collect persons with their velocity
        persons = {}
        for gid, item in display_items.items():
            if item.get("class_id", 0) == PERSON:
                persons[gid] = item

        current_active = set()

        for p_gid, p_item in persons.items():
            p_box = p_item["box"]
            p_vel = p_item.get("vel") or [0, 0]
            lat_vel = abs(float(p_vel[0])) if p_vel else 0.0

            rz = _reach_zone(p_box)

            for zone in shelf_zones:
                zone_name = zone[0]
                zone_box = (zone[1], zone[2], zone[3], zone[4])

                if not _box_intersects(rz, zone_box):
                    # No overlap — decay counter
                    key = (p_gid, zone_name)
                    if key in self._overlap_counters:
                        self._overlap_counters[key] = max(0,
                            self._overlap_counters[key] - 1)
                    continue

                # Velocity gate: walking past should not trigger
                if lat_vel > _MAX_WALK_VELOCITY:
                    continue

                key = (p_gid, zone_name)
                self._overlap_counters[key] += 1
                current_active.add(key)

                if self._overlap_counters[key] >= self._confirm_frames:
                    if key not in self._active_interactions:
                        ev = InteractionEvent(
                            person_gid=p_gid,
                            zone_name=zone_name,
                            zone_box=zone_box,
                            reach_box=rz,
                            person_box=list(p_box),
                            frame=frame_count,
                        )
                        ev.confirmed = True
                        self._active_interactions[key] = ev
                        print(f"[INTERACT] person_gid={p_gid} zone={zone_name} "
                              f"event=SHELF_INTERACTION_START frame={frame_count}")
                    else:
                        # Update reach box to current position
                        ev = self._active_interactions[key]
                        ev.reach_box = rz
                        ev.person_box = list(p_box)

        # Mark ended interactions
        ended = self._prev_active - current_active
        for key in ended:
            if key in self._active_interactions:
                ev = self._active_interactions[key]
                p_gid, zone_name = key
                duration = frame_count - ev.start_frame
                print(f"[INTERACT] person_gid={p_gid} zone={zone_name} "
                      f"event=SHELF_INTERACTION_END duration={duration}f "
                      f"frame={frame_count}")
                del self._active_interactions[key]
            self._overlap_counters.pop(key, None)

        self._prev_active = current_active

        return list(self._active_interactions.values())

    def get_active_for_person(self, person_gid):
        """Get all active interactions for a specific person."""
        return [ev for key, ev in self._active_interactions.items()
                if key[0] == person_gid]

    def get_reach_zone(self, person_box):
        """Public utility: compute reach zone for a person box."""
        return _reach_zone(person_box)
