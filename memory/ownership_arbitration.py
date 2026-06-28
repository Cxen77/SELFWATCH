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
        # Exclude THINKING tracks from arbitration. Arbitration should only
        # merge overlapping LIVE detections (STATE_ACTIVE). Predicted ghosts
        # drifting together should not trigger duplicate suppression.
        active_gids = [gid for gid, data in display.items() if data[1] == STATE_ACTIVE]
        overlaps = []
        for i in range(len(active_gids)):
            for j in range(i + 1, len(active_gids)):
                a, b = active_gids[i], active_gids[j]
                # Skip cross-class pairs: a person overlapping a bottle/bag
                # is expected physical reality, not a tracking duplicate.
                cid_a = display[a][5].get("class_id", 0) if isinstance(display[a][5], dict) else 0
                cid_b = display[b][5].get("class_id", 0) if isinstance(display[b][5], dict) else 0
                if cid_a != cid_b:
                    continue
                iou = self._iou(display[a][0], display[b][0])
                if iou > self.DUPLICATE_IOU_THRESH:
                    overlaps.append((a, b, iou))
        return overlaps

    # ── Simple Seniority Resolution ──────────────────────────────────

    def _resolve(self, gid_a, gid_b, display):
        """
        Resolve duplicate visual identity by prioritizing absolute maturity 
        and high-confidence anchor matches.
        
        Priority:
          1. Immunity Gate: GID with anchor_similarity > 0.50 wins over one without.
          2. Absolute Maturity (total hits): Older global identity wins.
          3. Fallback: ACTIVE vs THINKING state.
          4. Fallback: Short-term stability counters.
        """
        key = (min(gid_a, gid_b), max(gid_a, gid_b))
        
        state_a, state_b = display[gid_a][1], display[gid_b][1]
        hits_a = display[gid_a][5].get("hits", 0)
        hits_b = display[gid_b][5].get("hits", 0)
        anchor_sim_a = display[gid_a][5].get("anchor_sim", 0.0)
        anchor_sim_b = display[gid_b][5].get("anchor_sim", 0.0)

        # 1. Immunity Gate
        if anchor_sim_a > 0.50 and anchor_sim_b <= 0.50:
            winner, loser = gid_a, gid_b
            reason = f"immunity_gate({anchor_sim_a:.2f})"
        elif anchor_sim_b > 0.50 and anchor_sim_a <= 0.50:
            winner, loser = gid_b, gid_a
            reason = f"immunity_gate({anchor_sim_b:.2f})"
        
        # 2. Absolute Maturity
        elif hits_a > hits_b:
            winner, loser = gid_a, gid_b
            reason = f"maturity({hits_a}vs{hits_b})"
        elif hits_b > hits_a:
            winner, loser = gid_b, gid_a
            reason = f"maturity({hits_b}vs{hits_a})"
            
        # 3. Fallback to state
        elif state_a == STATE_ACTIVE and state_b == STATE_THINKING:
            winner, loser = gid_a, gid_b
            reason = "state(ACTIVEvsTHINKING)"
        elif state_b == STATE_ACTIVE and state_a == STATE_THINKING:
            winner, loser = gid_b, gid_a
            reason = "state(ACTIVEvsTHINKING)"
            
        # 4. Fallback to stability
        else:
            stab_a = self._stability_counters.get(gid_a, 0)
            stab_b = self._stability_counters.get(gid_b, 0)
            if stab_a >= stab_b:
                winner, loser = gid_a, gid_b
                reason = f"seniority({stab_a}vs{stab_b})"
            else:
                winner, loser = gid_b, gid_a
                reason = f"seniority({stab_b}vs{stab_a})"

        self._visual_owners[key] = winner

        if gid_a == 10 or gid_b == 10:
            print(f"[GID10 ARBITRATION] frame={self._frame_count} "
                  f"gid_a={gid_a} gid_b={gid_b} "
                  f"state_a={state_a} state_b={state_b} "
                  f"total_hits_a={hits_a} total_hits_b={hits_b} "
                  f"anchor_sim_a={anchor_sim_a} anchor_sim_b={anchor_sim_b} "
                  f"reason={reason} winner={winner} loser={loser}")

        return winner, loser, reason

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
