"""
Rendered Box Continuity Metric — measures what a HUMAN VIEWER sees on screen.

This metric is completely gid-agnostic. It tracks rendered bounding boxes
frame-to-frame by IoU matching, detecting when a box at a given spatial position
suddenly has a different identity than it should based on visual continuity.

Key insight: a human watching the video doesn't know internal tracker gids.
They track boxes by position, size, and visual appearance. When a box that was
being tracked by position suddenly shows a completely different ID label, that's
a visible switch — regardless of what the tracker internally believes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def box_center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class BoxContinuityMetric:
    """
    Tracks rendered box continuity across frames.

    Detection categories:
      - MATCH: box tracked continuously (same entity)
      - BIRTH: genuinely new box (no predecessor)
      - SWITCH: box appears where a different entity was last frame
      - DEATH: previously tracked box has no successor this frame
    """

    def __init__(self, iou_threshold: float = 0.20, max_death_memory: int = 10):
        self.iou_threshold = iou_threshold
        self.max_death_memory = max_death_memory

        self.switch_count = 0
        self.birth_count = 0
        self.death_count = 0
        self.id_change_count = 0  # box matches spatially but gid changed

        self.switch_events: List[dict] = []
        self.last_boxes: List[dict] = []  # [{gid, box, center, died_frames_ago}]

        self._raw_events: List[dict] = []  # per-frame diagnostic log

    def update(self, frame_idx: int, visible_objects: List[dict]):
        """
        visible_objects: list of dicts with 'global_id', 'bbox' [x1,y1,x2,y2]
        """
        current_entities = []
        for obj in visible_objects:
            bbox = obj.get('bbox')
            if bbox is None:
                continue
            current_entities.append({
                'gid': obj['global_id'],
                'box': list(bbox),
                'center': box_center(bbox),
            })

        # ── Match current to previous by IoU (greedy, best-first) ──
        matches = []  # [(prev_idx, curr_idx, iou, gid_changed)]
        prev_available = set(range(len(self.last_boxes)))
        curr_available = set(range(len(current_entities)))

        # Score all pairs
        pair_scores = []
        for pi in prev_available:
            pb = self.last_boxes[pi]
            # Skip very stale entries for matching purposes
            if pb.get('died_frames_ago', 0) > self.max_death_memory:
                continue
            for ci in curr_available:
                ce = current_entities[ci]
                iou = box_iou(pb['box'], ce['box'])
                if iou >= self.iou_threshold:
                    pair_scores.append((iou, pi, ci, pb['gid'] != ce['gid']))

        # Greedy assignment (highest IoU first)
        pair_scores.sort(key=lambda x: -x[0])
        for iou, pi, ci, gid_changed in pair_scores:
            if pi in prev_available and ci in curr_available:
                matches.append((pi, ci, iou, gid_changed))
                prev_available.discard(pi)
                curr_available.discard(ci)

        # ── Classify events ─────────────────────────────────────────
        # BIRTHS: current entities not matched to any previous
        for ci in curr_available:
            ce = current_entities[ci]
            # Check if this new box appears near where a recently dead box was
            # (different entity filling the same space → possible ID switch)
            near_death_entry = None
            for pi, pb in enumerate(self.last_boxes):
                if pb.get('died_frames_ago', 0) > self.max_death_memory:
                    continue
                if pb.get('gid') == ce['gid']:
                    continue  # Same gid, normal reappearance
                iou = box_iou(pb['box'], ce['box'])
                if iou >= self.iou_threshold:
                    near_death_entry = (pi, pb, iou)
                    break

            if near_death_entry:
                pi, pb, iou = near_death_entry
                # Box in essentially the same position, different gid → SWITCH
                self.switch_count += 1
                self.switch_events.append({
                    'frame': frame_idx,
                    'type': 'box_continuity_switch',
                    'old_id': pb['gid'],
                    'new_id': ce['gid'],
                    'iou': round(iou, 4),
                    'position': ce['center'],
                })
            else:
                self.birth_count += 1

        # GID changes on matched pairs (same spatial position, different display ID)
        for pi, ci, iou, gid_changed in matches:
            if gid_changed:
                prev_is_dead = self.last_boxes[pi].get('died_frames_ago', 0) > 0
                if prev_is_dead:
                    # Previous box had died — this is a continuity switch, not just a display change
                    self.switch_count += 1
                    self.switch_events.append({
                        'frame': frame_idx,
                        'type': 'box_continuity_switch',
                        'old_id': self.last_boxes[pi]['gid'],
                        'new_id': current_entities[ci]['gid'],
                        'iou': round(iou, 4),
                        'position': current_entities[ci]['center'],
                    })
                else:
                    self.id_change_count += 1
                    self.switch_events.append({
                        'frame': frame_idx,
                        'type': 'display_id_change',
                        'old_id': self.last_boxes[pi]['gid'],
                        'new_id': current_entities[ci]['gid'],
                        'iou': round(iou, 4),
                        'position': current_entities[ci]['center'],
                    })

        # DEATHS: previous boxes not matched to any current
        for pi in prev_available:
            pb = self.last_boxes[pi]
            new_died = pb.get('died_frames_ago', 0) + 1
            pb['died_frames_ago'] = new_died
            if new_died == 1:
                self.death_count += 1

        # ── Update state for next frame ─────────────────────────────
        new_last = []
        # Carry forward matched entities (with latest box info)
        for pi, ci, iou, gid_changed in matches:
            new_last.append({
                'gid': current_entities[ci]['gid'],
                'box': current_entities[ci]['box'],
                'center': current_entities[ci]['center'],
                'died_frames_ago': 0,
            })

        # Add new births so they can be tracked in subsequent frames
        for ci in curr_available:
            ce = current_entities[ci]
            new_last.append({
                'gid': ce['gid'],
                'box': ce['box'],
                'center': ce['center'],
                'died_frames_ago': 0,
            })

        # Keep dying entries (for near-death switch detection)
        for pi in prev_available:
            pb = self.last_boxes[pi]
            if pb.get('died_frames_ago', 0) <= self.max_death_memory:
                new_last.append(pb)

        self.last_boxes = new_last

        # ── Store raw diagnostic ────────────────────────────────────
        if self.switch_events and self.switch_events[-1]['frame'] == frame_idx:
            self._raw_events.append({
                'frame': frame_idx,
                'n_current': len(current_entities),
                'n_prev': len(self.last_boxes),
                'n_matches': len(matches),
                'n_births': len(curr_available),
                'n_deaths': len(prev_available),
            })

    def get_metrics(self) -> dict:
        return {
            'box_continuity_switches': self.switch_count,
            'box_births': self.birth_count,
            'box_deaths': self.death_count,
            'display_id_changes': self.id_change_count,
            'total_visible_switches': self.switch_count + self.id_change_count,
            'switch_events': self.switch_events
        }
