"""
Persistence Engine — Parent-Child object permanence and concealment reasoning.

When an unknown merchandise item is promoted to a tracked GID:
  1. It is bound as a "child" to the person who picked it up (parent GID).
  2. If the item's visual track drops out while spatially contained inside
     the parent's bounding box, the item transitions to CONCEALED state.
  3. A concealed item inherits the parent's spatial trajectory.
  4. If the parent crosses an exit zone with concealed items, theft is flagged.

State machine per item:
  SHELF -> PICKUP -> CARRIED -> CONCEALED -> EXITED (THEFT)
                  |-> RETURNED (item back on shelf)

Forensic logging: [PERSIST] on every state transition.
"""

import math
import time as _time
from collections import defaultdict

# ── Item States ──
STATE_SHELF = "ON_SHELF"
STATE_PICKUP = "PICKUP"
STATE_CARRIED = "CARRIED"
STATE_LATENT = "LATENT_HOLDING"  # Item near chest, might be returned
STATE_CONCEALED = "CONCEALED"
STATE_RETURNED = "RETURNED"
STATE_EXITED = "EXITED"

# How many frames of visual dropout inside parent bbox before CONCEALED
CONCEAL_CONFIRM_FRAMES = 12
# Minimum pixel separation from origin shelf before concealment can lock
SHELF_SEPARATION_PX = 80
# How many frames in LATENT before upgrading to CONCEALED
LATENT_TO_CONCEAL_FRAMES = 20
# Exit zone: bottom N% of frame
EXIT_ZONE_FRACTION = 0.15

from .interaction_engine import _reach_zone, _box_intersects


def _iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _box_contains(parent, child, threshold=0.5):
    """Check if child box is at least threshold-fraction inside parent box."""
    ix1 = max(parent[0], child[0])
    iy1 = max(parent[1], child[1])
    ix2 = min(parent[2], child[2])
    iy2 = min(parent[3], child[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    child_area = max(1, (child[2] - child[0]) * (child[3] - child[1]))
    return (inter / child_area) >= threshold


class TrackedItem:
    """A promoted merchandise item with parent-child binding."""
    __slots__ = [
        "item_gid", "candidate_id", "parent_gid",
        "state", "box", "origin_zone", "origin_box",
        "last_visible_frame", "last_visible_box",
        "conceal_counter", "latent_counter",
        "birth_frame", "state_history",
        "suspicion_contributed",
    ]

    def __init__(self, item_gid, candidate_id, parent_gid,
                 box, origin_zone, origin_box, frame):
        self.item_gid = item_gid
        self.candidate_id = candidate_id
        self.parent_gid = parent_gid
        self.state = STATE_PICKUP
        self.box = list(box)
        self.origin_zone = origin_zone
        self.origin_box = list(origin_box)
        self.last_visible_frame = frame
        self.last_visible_box = list(box)
        self.conceal_counter = 0
        self.latent_counter = 0
        self.birth_frame = frame
        self.state_history = [(frame, STATE_PICKUP)]
        self.suspicion_contributed = False

    def _transition(self, new_state, frame):
        if self.state != new_state:
            old = self.state
            self.state = new_state
            self.state_history.append((frame, new_state))
            print(f"[PERSIST] item_gid={self.item_gid} parent={self.parent_gid} "
                  f"transition={old}->{new_state} frame={frame}")


class PersistenceEngine:
    """
    Manages parent-child object permanence.

    Integrates with:
    - CandidateManager (receives promoted items)
    - InteractionEngine (knows which persons are at shelves)
    - Display items (knows current visible tracks)
    - Shelf zones (knows zone geometry)
    """

    def __init__(self, frame_h=540):
        self._items = {}  # item_gid -> TrackedItem
        self._frame_h = frame_h
        self._theft_alerts = []  # list of alert dicts
        self._next_item_gid = 5000  # High range to avoid GID collision

    def _alloc_gid(self):
        gid = self._next_item_gid
        self._next_item_gid += 1
        return gid

    def register_promoted(self, candidate, frame_count):
        """
        Register a promoted CandidateItem as a tracked merchandise entity.

        Args:
            candidate: CandidateItem from CandidateManager
            frame_count: current frame number

        Returns:
            item_gid assigned to this item
        """
        item_gid = self._alloc_gid()
        candidate.promoted_gid = item_gid

        item = TrackedItem(
            item_gid=item_gid,
            candidate_id=candidate.candidate_id,
            parent_gid=candidate.person_gid,
            box=candidate.box,
            origin_zone=candidate.origin_zone,
            origin_box=candidate.origin_box,
            frame=frame_count,
        )
        self._items[item_gid] = item

        print(f"[PERSIST] item_gid={item_gid} candidate={candidate.candidate_id} "
              f"parent={candidate.person_gid} zone={candidate.origin_zone} "
              f"event=REGISTERED frame={frame_count}")

        return item_gid

    def update(self, display_items, shelf_zones, candidate_items,
               frame_count):
        """
        Main update loop. Processes all tracked items, transitions states.

        Args:
            display_items: dict gid -> {"box", "vel", "class_id", ...}
            shelf_zones: list of (zone_name, x1, y1, x2, y2)
            candidate_items: list of active CandidateItem objects
            frame_count: current frame number

        Returns:
            dict with theft alerts and item states
        """
        self._theft_alerts = []

        # Build lookup for candidate boxes (for items still being tracked
        # by CandidateManager's IoU tracker)
        cand_by_gid = {}
        for cand in candidate_items:
            if cand.promoted and cand.promoted_gid:
                cand_by_gid[cand.promoted_gid] = cand

        for item_gid, item in list(self._items.items()):
            if item.state in (STATE_RETURNED, STATE_EXITED):
                continue

            parent_item = display_items.get(item.parent_gid)
            if parent_item is None:
                # Parent person left view — if item is concealed, this is theft
                if item.state == STATE_CONCEALED:
                    item._transition(STATE_EXITED, frame_count)
                    self._fire_alert(item, frame_count, reason="parent_gone")
                continue

            parent_box = parent_item["box"]

            # Update item box from candidate tracker if still being tracked
            cand = cand_by_gid.get(item_gid)
            item_visible = False
            if cand and cand.lost_frames == 0:
                item.box = list(cand.box)
                item.last_visible_frame = frame_count
                item.last_visible_box = list(cand.box)
                item_visible = True
                item.conceal_counter = 0
                item.latent_counter = 0

            # Also check: does this item appear as a tracked detection in
            # display_items (it might have been picked up by RF-DETR at low conf)?
            if not item_visible:
                # Check if any non-person display item is near this item's
                # last known position
                for gid, d_item in display_items.items():
                    if d_item.get("class_id", 0) == 0:  # Skip persons
                        continue
                    if _iou(item.last_visible_box, d_item["box"]) > 0.3:
                        item.box = list(d_item["box"])
                        item.last_visible_frame = frame_count
                        item.last_visible_box = list(d_item["box"])
                        item_visible = True
                        item.conceal_counter = 0
                        item.latent_counter = 0
                        break

            # ── State transitions ──

            if item.state == STATE_PICKUP:
                # Check if person has moved away from shelf
                origin_center = _center(item.origin_box)
                person_center = _center(parent_box)
                shelf_dist = _dist(origin_center, person_center)
                if shelf_dist > SHELF_SEPARATION_PX:
                    item._transition(STATE_CARRIED, frame_count)
                elif not item_visible:
                    # Item dropped out near shelf — might be returned
                    item._transition(STATE_RETURNED, frame_count)
                    print(f"[PERSIST] item_gid={item_gid} event=RETURNED_NEAR_SHELF "
                          f"frame={frame_count}")

            elif item.state == STATE_CARRIED:
                if item_visible:
                    # Still tracking the item — check if returned to shelf
                    item_center = _center(item.box)
                    rz = _reach_zone(parent_box)
                    # Only return if the item is on the shelf AND not in the person's reach zone
                    if not _box_intersects(item.box, rz):
                        for zone in shelf_zones:
                            zone_box = (zone[1], zone[2], zone[3], zone[4])
                            if (zone_box[0] <= item_center[0] <= zone_box[2] and
                                    zone_box[1] <= item_center[1] <= zone_box[3]):
                                item._transition(STATE_RETURNED, frame_count)
                                print(f"[PERSIST] item_gid={item_gid} "
                                      f"event=RETURNED_TO_SHELF zone={zone[0]} "
                                      f"frame={frame_count}")
                                break
                else:
                    # Item visual dropped out
                    # Check if last known position was inside parent bbox
                    if _box_contains(parent_box, item.last_visible_box, 0.3):
                        item.conceal_counter += 1
                        if item.conceal_counter >= CONCEAL_CONFIRM_FRAMES:
                            # Check shelf separation
                            origin_center = _center(item.origin_box)
                            person_center = _center(parent_box)
                            if _dist(origin_center, person_center) > SHELF_SEPARATION_PX:
                                item._transition(STATE_CONCEALED, frame_count)
                            else:
                                item._transition(STATE_LATENT, frame_count)
                    else:
                        # Lost and NOT inside parent — just lost
                        item.conceal_counter += 1
                        if item.conceal_counter > CONCEAL_CONFIRM_FRAMES * 3:
                            # Really gone, remove
                            print(f"[PERSIST] item_gid={item_gid} event=LOST "
                                  f"frame={frame_count}")
                            item._transition(STATE_RETURNED, frame_count)

            elif item.state == STATE_LATENT:
                if item_visible:
                    # Item reappeared — go back to carried
                    item._transition(STATE_CARRIED, frame_count)
                    item.latent_counter = 0
                else:
                    item.latent_counter += 1
                    # Check if person moved away from shelf
                    origin_center = _center(item.origin_box)
                    person_center = _center(parent_box)
                    if (_dist(origin_center, person_center) > SHELF_SEPARATION_PX and
                            item.latent_counter >= LATENT_TO_CONCEAL_FRAMES):
                        item._transition(STATE_CONCEALED, frame_count)

            elif item.state == STATE_CONCEALED:
                if item_visible:
                    # Item reappeared! Un-conceal.
                    item._transition(STATE_CARRIED, frame_count)
                    item.conceal_counter = 0
                else:
                    # Concealed — inherit parent position
                    item.box = list(parent_box)

                    # Check exit zone
                    py_bottom = parent_box[3]
                    exit_y = self._frame_h * (1.0 - EXIT_ZONE_FRACTION)
                    if py_bottom > exit_y:
                        item._transition(STATE_EXITED, frame_count)
                        self._fire_alert(item, frame_count, reason="exit_zone")

                    # Also check if person returned to the shelf zone
                    person_center = _center(parent_box)
                    for zone in shelf_zones:
                        zone_box = (zone[1], zone[2], zone[3], zone[4])
                        if (zone_box[0] <= person_center[0] <= zone_box[2] and
                                zone_box[1] <= person_center[1] <= zone_box[3]):
                            # Person returned to shelf — transition to latent
                            # (they might be putting item back)
                            item._transition(STATE_LATENT, frame_count)
                            break

        return {
            "theft_alerts": list(self._theft_alerts),
            "items": {gid: {
                "state": it.state,
                "box": it.box,
                "parent_gid": it.parent_gid,
                "origin_zone": it.origin_zone,
                "concealed": it.state == STATE_CONCEALED,
            } for gid, it in self._items.items()
                if it.state not in (STATE_RETURNED, STATE_EXITED)},
        }

    def _fire_alert(self, item, frame_count, reason=""):
        alert = {
            "alert": "THEFT",
            "item_gid": item.item_gid,
            "person_gid": item.parent_gid,
            "origin_zone": item.origin_zone,
            "reason": reason,
            "frame": frame_count,
            "state_history": list(item.state_history),
        }
        self._theft_alerts.append(alert)
        if not item.suspicion_contributed:
            item.suspicion_contributed = True

        print(f"\n{'='*60}")
        print(f"[THEFT ALERT] UNKNOWN MERCHANDISE THEFT DETECTED")
        print(f"  Item GID:     {item.item_gid}")
        print(f"  Parent GID:   {item.parent_gid}")
        print(f"  Origin Zone:  {item.origin_zone}")
        print(f"  Reason:       {reason}")
        print(f"  Frame:        {frame_count}")
        print(f"  State History:")
        for f, s in item.state_history:
            print(f"    F#{f}: {s}")
        print(f"{'='*60}\n")

    def get_concealed_for_person(self, person_gid):
        """Return all concealed items bound to a person."""
        return [it for it in self._items.values()
                if it.parent_gid == person_gid and it.state == STATE_CONCEALED]

    def get_all_items(self):
        """Return all tracked items."""
        return dict(self._items)

    def get_item(self, item_gid):
        return self._items.get(item_gid)

    def close(self):
        """Print final report."""
        active = [it for it in self._items.values()
                  if it.state not in (STATE_RETURNED,)]
        concealed = [it for it in self._items.values()
                     if it.state == STATE_CONCEALED]
        exited = [it for it in self._items.values()
                  if it.state == STATE_EXITED]

        print(f"\n{'='*60}")
        print(f"  PERSISTENCE ENGINE — Final Report")
        print(f"{'='*60}")
        print(f"  Total items tracked: {len(self._items)}")
        print(f"  Active:     {len(active)}")
        print(f"  Concealed:  {len(concealed)}")
        print(f"  Exited:     {len(exited)}")
        print(f"  Alerts:     {len(self._theft_alerts)}")
        for it in self._items.values():
            print(f"  Item GID={it.item_gid} parent={it.parent_gid} "
                  f"state={it.state} zone={it.origin_zone}")
        print(f"{'='*60}")
