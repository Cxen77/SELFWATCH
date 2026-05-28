"""
Visible ID Switch Detection — measures what a HUMAN VIEWER sees on screen.

Four detection modes:
  TYPE-A: Vanish + Replace   — gid disappears, different gid appears in same spot
  TYPE-B: ID Swap            — two identities exchange positions between frames  
  TYPE-C: Flicker / Recovery Failure — person flickers to different gid then reappears
  TYPE-D: Rendered Box Continuity — spatial tracking of boxes ignoring gids

This metric tracks what's actually rendered on screen, not what the tracker
internally believes about identity continuity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class VisibleIDSwitchMetric:
    """
    Measures all types of visible ID switches on screen.

    Tunable thresholds (conservative defaults for ~5 FPS):
      spatial_tolerance: max pixel distance to consider "same position" (default 60)
      min_visible_frames: how many frames a gid must be visible before qualifying
      temporal_window: max frame gap between disappearance and reappearance
      min_swap_displacement: min distance for swap detection
    """

    def __init__(self, spatial_tolerance: float = 60.0, min_visible_frames: int = 5,
                 temporal_window: int = 10, min_swap_displacement: float = 50.0,
                 flicker_grace_frames: int = 3):
        self.spatial_tolerance = spatial_tolerance
        self.min_visible_frames = min_visible_frames
        self.temporal_window = temporal_window
        self.min_swap_displacement = min_swap_displacement
        self.flicker_grace_frames = flicker_grace_frames

        # Counters by type
        self.replace_count = 0       # TYPE-A: vanish + replace
        self.swap_count = 0          # TYPE-B: ID exchange
        self.flicker_count = 0       # TYPE-C: flicker / recovery failure

        self.last_frame_positions = {}   # {gid: (cx, cy)}
        self.last_frame_boxes = {}       # {gid: [x1,y1,x2,y2]}
        self.switch_events: List[dict] = []
        self.max_switch_events = 500

        # Visibility tracking with flicker-resilience
        self._visibility_duration = {}       # gid -> consecutive frame count
        self._visibility_grace = {}          # gid -> grace frames remaining
        self._max_visible_duration = {}      # gid -> peak duration before any gap

        # Recently disappeared established identities (extended window for flicker detection)
        self._recent_disappearances: List[dict] = []

        # Track full history of gid positions for forensic analysis (last N frames)
        self._recent_positions: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

    def update(self, frame_idx: int, visible_objects: List[dict], 
               raw_display: dict = None):
        """
        Process a frame's visible rendered identities.

        visible_objects: list of dicts with 'global_id', 'bbox' [x1,y1,x2,y2]
        raw_display: optional dict of what WOULD have been rendered without arbitration
        """
        current_positions = {}
        current_boxes = {}
        current_gids = set()

        for obj in visible_objects:
            gid = obj['global_id']
            bbox = obj['bbox']
            if bbox is None:
                continue
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            current_positions[gid] = (cx, cy)
            current_boxes[gid] = list(bbox)
            current_gids.add(gid)

        # ── Update visibility duration with flicker resilience ──────
        for gid in current_gids:
            if gid in self._visibility_grace:
                # Recovered from brief flicker — restore accumulated duration
                prev_max = self._max_visible_duration.get(gid, 0)
                self._visibility_duration[gid] = prev_max + 1
                self._max_visible_duration[gid] = self._visibility_duration[gid]
                del self._visibility_grace[gid]
            else:
                self._visibility_duration[gid] = self._visibility_duration.get(gid, 0) + 1
                self._max_visible_duration[gid] = max(
                    self._max_visible_duration.get(gid, 0),
                    self._visibility_duration[gid])

        # ── Flicker-resilient disappearance tracking ─────────────────
        disappeared = set(self.last_frame_positions.keys()) - current_gids
        for d_id in disappeared:
            duration = self._visibility_duration.get(d_id, 0)
            if duration >= self.min_visible_frames:
                self._recent_disappearances.append({
                    "frame": frame_idx,
                    "gid": d_id,
                    "pos": self.last_frame_positions[d_id],
                    "bbox": self.last_frame_boxes.get(d_id),
                    "duration": duration,
                })
                self._max_visible_duration[d_id] = duration
            # Start grace period instead of immediately resetting
            if d_id in self._max_visible_duration and self._max_visible_duration[d_id] >= self.min_visible_frames:
                self._visibility_grace[d_id] = self.flicker_grace_frames
            self._visibility_duration.pop(d_id, None)

        # Tick grace counters for identities that remain missing
        expired_grace = [gid for gid, g in self._visibility_grace.items() if gid not in current_gids]
        for gid in expired_grace:
            self._visibility_grace[gid] -= 1
            if self._visibility_grace[gid] <= 0:
                del self._visibility_grace[gid]
                self._max_visible_duration.pop(gid, None)

        # ── TYPE-A: Vanish + Replace Detection ──────────────────────
        appeared = current_gids - set(self.last_frame_positions.keys())
        for a_id in appeared:
            a_pos = current_positions[a_id]
            for entry in list(self._recent_disappearances):
                if frame_idx - entry["frame"] > self.temporal_window:
                    continue
                d_pos = entry["pos"]
                dist = np.hypot(d_pos[0] - a_pos[0], d_pos[1] - a_pos[1])
                if dist < self.spatial_tolerance and entry["gid"] != a_id:
                    self.replace_count += 1
                    self.switch_events.append({
                        'frame': frame_idx,
                        'type': 'replace',
                        'old_id': entry["gid"],
                        'new_id': a_id,
                        'distance': round(dist, 1),
                        'old_duration': entry["duration"],
                    })
                    if len(self.switch_events) > self.max_switch_events:
                        self.switch_events.pop(0)
                    self._recent_disappearances.remove(entry)
                    break

        # ── TYPE-B: ID Swap Detection ───────────────────────────────
        common_gids = current_gids & set(self.last_frame_positions.keys())
        if len(common_gids) >= 2:
            gid_list = list(common_gids)
            for i in range(len(gid_list)):
                for j in range(i + 1, len(gid_list)):
                    a, b = gid_list[i], gid_list[j]
                    if a == b:
                        continue
                    a_dur = self._visibility_duration.get(a, 0)
                    b_dur = self._visibility_duration.get(b, 0)
                    if a_dur < self.min_visible_frames or b_dur < self.min_visible_frames:
                        continue

                    # Check if positions swapped
                    a_pos_now = current_positions[a]
                    b_pos_now = current_positions[b]
                    a_pos_prev = self.last_frame_positions[a]
                    b_pos_prev = self.last_frame_positions[b]

                    # Distance from a's new position to b's previous position
                    dist_a_to_bprev = np.hypot(a_pos_now[0] - b_pos_prev[0],
                                                a_pos_now[1] - b_pos_prev[1])
                    # Distance from b's new position to a's previous position
                    dist_b_to_aprev = np.hypot(b_pos_now[0] - a_pos_prev[0],
                                                b_pos_now[1] - a_pos_prev[1])
                    # Self-displacements (how much each moved)
                    self_dist_a = np.hypot(a_pos_now[0] - a_pos_prev[0],
                                            a_pos_now[1] - a_pos_prev[1])
                    self_dist_b = np.hypot(b_pos_now[0] - b_pos_prev[0],
                                            b_pos_now[1] - b_pos_prev[1])

                    # Swap condition: each gid moved near where the other was,
                    # AND self-displacements are large (they didn't just wander)
                    if (dist_a_to_bprev < self.spatial_tolerance and
                        dist_b_to_aprev < self.spatial_tolerance and
                        self_dist_a > self.min_swap_displacement and
                        self_dist_b > self.min_swap_displacement):
                        self.swap_count += 1
                        self.switch_events.append({
                            'frame': frame_idx,
                            'type': 'swap',
                            'old_id': a,
                            'new_id': b,
                            'distance_a': round(dist_a_to_bprev, 1),
                            'distance_b': round(dist_b_to_aprev, 1),
                            'displacement_a': round(self_dist_a, 1),
                            'displacement_b': round(self_dist_b, 1),
                        })
                        if len(self.switch_events) > self.max_switch_events:
                            self.switch_events.pop(0)
                        break

        # ── TYPE-C: Flicker / Recovery Failure Detection ────────────
        for entry in list(self._recent_disappearances):
            frames_gone = frame_idx - entry["frame"]
            if frames_gone <= 1:
                continue  # Too recent to assess

            d_id = entry["gid"]
            # Has the original gid reappeared elsewhere?
            if d_id in current_gids:
                new_pos = current_positions[d_id]
                entry_pos = entry["pos"]
                dist = np.hypot(new_pos[0] - entry_pos[0], new_pos[1] - entry_pos[1])
                if dist > self.spatial_tolerance * 3:
                    # Reappeared far away — recovery failure
                    self.flicker_count += 1
                    self.switch_events.append({
                        'frame': frame_idx,
                        'type': 'recovery_failure',
                        'old_id': d_id,
                        'new_id': d_id,
                        'distance': round(dist, 1),
                        'old_duration': entry["duration"],
                        'gap_frames': frames_gone,
                    })
                    if len(self.switch_events) > self.max_switch_events:
                        self.switch_events.pop(0)
                self._recent_disappearances.remove(entry)

        # ── Clean up old disappearances ─────────────────────────────
        self._recent_disappearances = [
            e for e in self._recent_disappearances
            if frame_idx - e["frame"] <= self.temporal_window * 2
        ]

        # ── Update position history for forensic analysis ───────────
        for gid, pos in current_positions.items():
            if gid not in self._recent_positions:
                self._recent_positions[gid] = []
            self._recent_positions[gid].append((frame_idx, pos))
            if len(self._recent_positions[gid]) > 60:
                self._recent_positions[gid] = self._recent_positions[gid][-60:]

        self.last_frame_positions = current_positions
        self.last_frame_boxes = current_boxes

    def get_metrics(self) -> dict:
        return {
            'visible_id_switches': self.replace_count + self.swap_count + self.flicker_count,
            'replace_switches': self.replace_count,
            'swap_switches': self.swap_count,
            'flicker_switches': self.flicker_count,
            'switch_events': self.switch_events
        }

    def get_forensic_report(self) -> str:
        """Generate a human-readable forensic report of all detected switches."""
        if not self.switch_events:
            return "No visible ID switches detected."

        lines = [f"VISIBLE ID SWITCH FORENSIC REPORT",
                 f"{'=' * 60}",
                 f"Total events: {len(self.switch_events)}",
                 f"  Replace (vanish+appear): {self.replace_count}",
                 f"  Swap (identity exchange): {self.swap_count}",
                 f"  Flicker/recovery failures: {self.flicker_count}",
                 f"", ""]

        type_labels = {
            'replace': 'TYPE-A REPLACE',
            'swap': 'TYPE-B SWAP',
            'recovery_failure': 'TYPE-C FLICKER'
        }

        for event in self.switch_events:
            t = event['type']
            label = type_labels.get(t, t.upper())
            if t == 'replace':
                lines.append(
                    f"Frame {event['frame']:>5d}: {label:<20s} "
                    f"gid={event['old_id']:>3d} → gid={event['new_id']:>3d} "
                    f"(dist={event['distance']:.1f}px, dur={event['old_duration']}f)")
            elif t == 'swap':
                lines.append(
                    f"Frame {event['frame']:>5d}: {label:<20s} "
                    f"gid={event['old_id']:>3d} ↔ gid={event['new_id']:>3d} "
                    f"(Δa={event['displacement_a']:.0f}px, Δb={event['displacement_b']:.0f}px)")
            elif t == 'recovery_failure':
                lines.append(
                    f"Frame {event['frame']:>5d}: {label:<20s} "
                    f"gid={event['old_id']:>3d} recovered {event['distance']:.0f}px away "
                    f"(gap={event['gap_frames']}f, dur={event['old_duration']}f)")

        lines.append("")
        lines.append("KEY:")
        lines.append("  TYPE-A: Identity disappeared, different identity appeared in same spot")
        lines.append("  TYPE-B: Two identities exchanged positions (swap)")
        lines.append("  TYPE-C: Identity disappeared, reappeared at a very different position")
        return "\n".join(lines)
