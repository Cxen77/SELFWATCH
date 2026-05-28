"""
Ownership Arbitration — Cognitive Persistence

Minimal duplicate resolution layer. No competition, no scoring, no challengers.

Core principle:
  "This is person X until there is overwhelming evidence otherwise."

This module does exactly two things:
  1. Detect when two boxes visually overlap (potential duplicate)
  2. Keep the senior identity and suppress the junior one

There is NO:
  - Dominance scoring
  - Confidence computation
  - Challenger frame counting
  - Sandbox testing
  - Visual lock thresholds
  - Hysteresis chains

Ownership changes only happen through the natural identity lifecycle
(ACTIVE → THINKING → PHANTOM → WARM → recovered), never through
arbitration-level competition.
"""

import numpy as np
from collections import defaultdict


STATE_ACTIVE = 0
STATE_THINKING = 1


class OwnershipArbitrationLayer:
    """
    Minimal duplicate suppression layer.

    When two identities spatially overlap, keeps the established
    (older/more stable) one and suppresses the newcomer.
    No competition. No re-evaluation. Seniority wins.
    """

    DUPLICATE_IOU_THRESH = 0.20

    def __init__(self):
        # Visual owner memory: remembers who owns each overlap region
        # region_key -> owner_gid
        self._visual_owners = {}
        # Stability: consecutive frames visible
        self._stability_counters = defaultdict(int)
        self._frame_count = 0

    def arbitrate(self, display, frozen_gids=None, frame_count=0):
        """
        Resolve duplicate boxes by keeping established owner.

        Args:
            display: dict of gid -> (box, state, lid, vel, age, assoc_data)
            frozen_gids: set of currently frozen gids
            frame_count: current frame number

        Returns:
            cleaned_display, suppressed_list
        """
        self._frame_count = frame_count
        frozen_gids = frozen_gids or set()

        if len(display) <= 1:
            self._update_stability(display)
            return display, []

        # Find overlapping pairs
        overlaps = self._find_overlaps(display)

        # Resolve each: senior identity wins
        suppressed = []
        remove = set()

        for gid_a, gid_b, iou in overlaps:
            if gid_a in remove or gid_b in remove:
                continue

            winner, loser, reason = self._resolve(gid_a, gid_b, display)
            if loser is not None:
                remove.add(loser)
                suppressed.append((loser, reason))

        cleaned = {g: d for g, d in display.items() if g not in remove}
        self._update_stability(cleaned)
        self._cleanup_stale_owners()

        return cleaned, suppressed

    # ── Overlap Detection ────────────────────────────────────────────

    def _find_overlaps(self, display):
        gids = list(display.keys())
        overlaps = []
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                a, b = gids[i], gids[j]
                iou = self._iou(display[a][0], display[b][0])
                if iou > self.DUPLICATE_IOU_THRESH:
                    overlaps.append((a, b, iou))
        return overlaps

    # ── Simple Seniority Resolution ──────────────────────────────────

    def _resolve(self, gid_a, gid_b, display):
        """
        Keep the established owner. No scoring, no competition.

        Priority:
          1. If we've seen this pair before, keep the previous winner
          2. Otherwise: ACTIVE beats THINKING
          3. Otherwise: higher stability wins
          4. Otherwise: lower gid (older identity) wins
        """
        key = (min(gid_a, gid_b), max(gid_a, gid_b))

        # Already established an owner for this pair — keep them
        if key in self._visual_owners:
            owner = self._visual_owners[key]
            if owner in display:
                other = gid_b if owner == gid_a else gid_a
                return owner, other, "keep_established"
            else:
                # Previous owner gone, new one takes over
                other = gid_b if owner == gid_a else gid_a
                self._visual_owners[key] = other
                return other, owner, "owner_gone"

        # New overlap — pick winner by seniority
        state_a, state_b = display[gid_a][1], display[gid_b][1]
        stab_a = self._stability_counters.get(gid_a, 0)
        stab_b = self._stability_counters.get(gid_b, 0)

        if state_a == STATE_ACTIVE and state_b == STATE_THINKING:
            winner, loser = gid_a, gid_b
        elif state_b == STATE_ACTIVE and state_a == STATE_THINKING:
            winner, loser = gid_b, gid_a
        elif stab_a >= stab_b:
            winner, loser = gid_a, gid_b
        else:
            winner, loser = gid_b, gid_a

        self._visual_owners[key] = winner
        return winner, loser, f"seniority({stab_a}vs{stab_b})"

    # ── Stability Tracking ───────────────────────────────────────────

    def _update_stability(self, display):
        visible = set(display.keys())
        for gid in visible:
            self._stability_counters[gid] += 1
        gone = [g for g in self._stability_counters if g not in visible]
        for g in gone:
            self._stability_counters[g] = max(0, self._stability_counters[g] - 2)

    def _cleanup_stale_owners(self):
        stale = [k for k, v in self._visual_owners.items()
                 if self._stability_counters.get(v, 0) == 0]
        for k in stale:
            del self._visual_owners[k]

    # ── Query API ────────────────────────────────────────────────────

    def get_ownership_confidence(self, gid):
        """Compatibility stub — confidence is binary in simplified model."""
        return 1.0 if self._stability_counters.get(gid, 0) > 0 else 0.0

    def get_shadow_hypotheses(self):
        """Compatibility stub — no shadow system in simplified model."""
        return {}

    def get_sandbox_candidates(self):
        """Compatibility stub — no sandbox in simplified model."""
        return {}

    @staticmethod
    def _iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / (area_a + area_b - inter + 1e-6)


class TrackAwareNMS:
    """
    Track-Aware Non-Maximum Suppression.
    Prevents spawning duplicate identities near existing active tracks.
    """

    @staticmethod
    def suppress_duplicate_births(new_detections, active_tracks, iou_thresh=0.30):
        if not active_tracks or not new_detections:
            return new_detections, []

        filtered = []
        suppressed = []

        for idx, (det_box, det_score) in enumerate(new_detections):
            is_dup = False
            for trk_box, trk_gid, trk_age in active_tracks:
                iou = TrackAwareNMS._iou(det_box, trk_box)
                if iou > iou_thresh and trk_age > 5:
                    is_dup = True
                    break
            if is_dup:
                suppressed.append(idx)
            else:
                filtered.append((det_box, det_score))

        return filtered, suppressed

    @staticmethod
    def _iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / (area_a + area_b - inter + 1e-6)
