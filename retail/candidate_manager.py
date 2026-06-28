"""
Candidate Manager — Two-tier temporary unknown merchandise tracking.

Tier 1 (Candidates): Lightweight IoU-tracked bounding boxes that have
not yet been promoted to full GIDs. Cheap, disposable, camera-local.

Tier 2 (Promoted Items): Candidates that survived enough frames and
demonstrated real physical displacement from a shelf zone. These get
assigned a real GID and are bound to a parent Person.

No ReID is used for candidates — only pure IoU/Kalman tracking.
Forensic logging: [CANDIDATE] on every lifecycle event.
"""

import math
import time as _time
from collections import defaultdict


# Minimum frames a candidate must survive before promotion
PROMOTE_MIN_FRAMES = 5
# Maximum distance (px) to associate candidate across frames via IoU
CANDIDATE_MAX_DIST = 60
# Minimum IoU to associate across frames
CANDIDATE_MIN_IOU = 0.15
# Maximum number of frames a candidate can be lost before removal
CANDIDATE_MAX_LOST = 8
# Minimum area (px^2) to consider a valid candidate
CANDIDATE_MIN_AREA = 150
# Maximum area — too large = probably a person or bag, not merchandise
CANDIDATE_MAX_AREA = 40000

_next_candidate_id = 9000  # Start IDs high to avoid collision with GIDs


def _iou(a, b):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
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


def _area(box):
    return max(0, (box[2] - box[0]) * (box[3] - box[1]))


class CandidateItem:
    """A Tier-1 tracked unknown merchandise candidate."""
    __slots__ = [
        "candidate_id", "box", "prev_box", "velocity",
        "origin_zone", "origin_box",
        "person_gid", "birth_frame", "age", "lost_frames",
        "promoted", "promoted_gid",
    ]

    def __init__(self, candidate_id, box, origin_zone, person_gid, frame):
        self.candidate_id = candidate_id
        self.box = list(box)
        self.prev_box = list(box)
        self.velocity = [0.0, 0.0]
        self.origin_zone = origin_zone
        self.origin_box = list(box)  # Where it first appeared (on shelf)
        self.person_gid = person_gid
        self.birth_frame = frame
        self.age = 0
        self.lost_frames = 0
        self.promoted = False
        self.promoted_gid = None

    def update_box(self, new_box):
        """Update position and compute velocity."""
        cx_old = (self.box[0] + self.box[2]) / 2
        cy_old = (self.box[1] + self.box[3]) / 2
        cx_new = (new_box[0] + new_box[2]) / 2
        cy_new = (new_box[1] + new_box[3]) / 2
        self.velocity = [cx_new - cx_old, cy_new - cy_old]
        self.prev_box = list(self.box)
        self.box = list(new_box)
        self.age += 1
        self.lost_frames = 0

    def predicted_box(self):
        """Simple velocity-based predicted position."""
        vx, vy = self.velocity
        return [
            self.box[0] + vx, self.box[1] + vy,
            self.box[2] + vx, self.box[3] + vy,
        ]


class CandidateManager:
    """
    Manages the lifecycle of unknown merchandise candidates.

    Call flow:
        1. process_proposals() — ingest new detection proposals from
           shelf change detector or low-confidence RF-DETR outputs
        2. Items are tracked frame-to-frame via IoU matching
        3. Candidates that survive PROMOTE_MIN_FRAMES are marked for promotion
        4. get_promoted() returns items ready for GID assignment
    """

    def __init__(self, promote_min_frames=PROMOTE_MIN_FRAMES):
        self._candidates = {}  # candidate_id -> CandidateItem
        self._promote_min_frames = promote_min_frames
        self._promoted_this_frame = []
        self._next_id = _next_candidate_id

    def _alloc_id(self):
        cid = self._next_id
        self._next_id += 1
        return cid

    def process_proposals(self, proposals, active_interactions, display_items,
                          frame_count):
        """
        Process new object proposals and update existing candidates.

        Args:
            proposals: list of {"box": [x1,y1,x2,y2], "confidence": float,
                       "zone": str, ...} from ShelfChangeDetector or
                       low-confidence RF-DETR detections
            active_interactions: list of InteractionEvent from InteractionEngine
            display_items: current frame's display_items dict (for person boxes)
            frame_count: current frame number

        Returns:
            list of active CandidateItem objects
        """
        self._promoted_this_frame = []

        # Build set of interacting person GIDs and their zones
        interacting = {}
        for ev in active_interactions:
            interacting.setdefault(ev.zone_name, []).append(ev.person_gid)

        # Filter proposals: only accept those in zones with active interaction
        valid_proposals = []
        for prop in proposals:
            zone = prop.get("zone", "")
            area = _area(prop["box"])
            if area < CANDIDATE_MIN_AREA or area > CANDIDATE_MAX_AREA:
                continue
            # Must be in a zone where someone is interacting
            persons_in_zone = interacting.get(zone, [])
            if not persons_in_zone:
                # Also accept if a person box overlaps the proposal
                # (item is being carried away from shelf)
                for gid, item in display_items.items():
                    if item.get("class_id", 0) == 0:  # PERSON
                        if _iou(prop["box"], item["box"]) > 0.05:
                            persons_in_zone = [gid]
                            break
            if not persons_in_zone:
                continue

            # Origin plane gating: reject items that appear from inside
            # a person's torso (personal belongings, not shelf merchandise)
            prop_center = _center(prop["box"])
            is_personal = False
            for p_gid in persons_in_zone:
                p_item = display_items.get(p_gid)
                if p_item:
                    p_box = p_item["box"]
                    # If proposal center is deep inside person torso, skip
                    p_cx = (p_box[0] + p_box[2]) / 2
                    p_cy = (p_box[1] + p_box[3]) / 2
                    p_w = p_box[2] - p_box[0]
                    p_h = p_box[3] - p_box[1]
                    # "Deep inside" = within central 50% of person box
                    inner_x1 = p_cx - p_w * 0.25
                    inner_x2 = p_cx + p_w * 0.25
                    inner_y1 = p_cy - p_h * 0.25
                    inner_y2 = p_cy + p_h * 0.25
                    if (inner_x1 < prop_center[0] < inner_x2 and
                            inner_y1 < prop_center[1] < inner_y2):
                        is_personal = True
                        break
            if is_personal:
                continue

            prop["_persons"] = persons_in_zone
            valid_proposals.append(prop)

        # ── Associate proposals to existing candidates ──
        matched_candidates = set()
        matched_proposals = set()

        # Build cost matrix: candidate predicted box vs proposal box
        for cid, cand in self._candidates.items():
            pred = cand.predicted_box()
            best_iou = 0
            best_idx = -1
            for i, prop in enumerate(valid_proposals):
                if i in matched_proposals:
                    continue
                iou_val = _iou(pred, prop["box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            # Also try center distance for small items
            if best_iou < CANDIDATE_MIN_IOU and best_idx == -1:
                pred_c = _center(pred)
                best_d = CANDIDATE_MAX_DIST
                for i, prop in enumerate(valid_proposals):
                    if i in matched_proposals:
                        continue
                    d = _dist(pred_c, _center(prop["box"]))
                    if d < best_d:
                        best_d = d
                        best_idx = i
                        best_iou = 0.01  # Flag as distance-matched

            if best_idx >= 0 and best_iou >= CANDIDATE_MIN_IOU:
                cand.update_box(valid_proposals[best_idx]["box"])
                matched_candidates.add(cid)
                matched_proposals.add(best_idx)

        # ── Mark unmatched candidates as lost ──
        for cid, cand in list(self._candidates.items()):
            if cid not in matched_candidates:
                cand.lost_frames += 1
                if cand.lost_frames > CANDIDATE_MAX_LOST:
                    print(f"[CANDIDATE] id={cid} event=EXPIRED "
                          f"age={cand.age}f zone={cand.origin_zone} "
                          f"frame={frame_count}")
                    del self._candidates[cid]

        # ── Create new candidates from unmatched proposals ──
        for i, prop in enumerate(valid_proposals):
            if i in matched_proposals:
                continue
            # Pick the closest interacting person
            best_person = prop["_persons"][0] if prop["_persons"] else None
            if best_person is None:
                continue

            cid = self._alloc_id()
            cand = CandidateItem(
                candidate_id=cid,
                box=prop["box"],
                origin_zone=prop.get("zone", "unknown"),
                person_gid=best_person,
                frame=frame_count,
            )
            self._candidates[cid] = cand
            print(f"[CANDIDATE] id={cid} event=CREATED "
                  f"zone={cand.origin_zone} person={best_person} "
                  f"box=({prop['box'][0]:.0f},{prop['box'][1]:.0f},"
                  f"{prop['box'][2]:.0f},{prop['box'][3]:.0f}) "
                  f"frame={frame_count}")

        # ── Check promotion ──
        for cid, cand in self._candidates.items():
            if cand.promoted:
                continue
            if cand.age >= self._promote_min_frames:
                cand.promoted = True
                self._promoted_this_frame.append(cand)
                print(f"[CANDIDATE] id={cid} event=PROMOTED "
                      f"age={cand.age}f zone={cand.origin_zone} "
                      f"person={cand.person_gid} frame={frame_count}")

        return list(self._candidates.values())

    def get_promoted(self):
        """Return candidates promoted this frame (ready for GID assignment)."""
        return list(self._promoted_this_frame)

    def get_active_candidates(self):
        """Return all non-promoted, living candidates."""
        return [c for c in self._candidates.values() if not c.promoted]

    def get_all(self):
        """Return all candidates (active + promoted)."""
        return list(self._candidates.values())

    def remove(self, candidate_id):
        """Remove a candidate (e.g., after GID assignment)."""
        self._candidates.pop(candidate_id, None)

    def update_promoted_box(self, candidate_id, new_box):
        """Update a promoted candidate's box from external tracker."""
        cand = self._candidates.get(candidate_id)
        if cand:
            cand.update_box(new_box)
