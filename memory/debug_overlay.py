"""
SELFWATCH — Debug Visualization Overlay

Lightweight optional on-screen overlays showing cognitive memory state
for each tracked person. Uses pure OpenCV drawing — zero allocations,
zero tensor ops. Skip entirely when disabled.
"""

import time
import cv2


# ── Overlay colors ───────────────────────────────────────────────────
_CLR_ACTIVE = (0, 255, 100)      # Green
_CLR_UNCERTAIN = (0, 200, 255)   # Orange
_CLR_WARM = (255, 180, 0)        # Cyan-blue
_CLR_LOCKED = (0, 0, 255)        # Red
_CLR_RESURRECT = (0, 255, 255)   # Yellow
_CLR_DIM = (160, 160, 160)       # Gray
_CLR_PANEL_BG = (30, 30, 30)     # Dark background


class DebugLayerManager:
    def __init__(self):
        self.layers = {
            # Base Categories
            "tracking": True,
            "motion": False,
            "association": False,
            "cognitive": True,
            "forensic": True,
            
            # Tracking sub-features
            "tracking_age": True,
            "tracking_ids": True,

            # Motion sub-features
            "motion_velocity": False,
            "motion_prediction": False,

            # Association sub-features
            "assoc_cost": False,
            "assoc_cbiou": False,
            "assoc_method": False,
            "assoc_ambiguity": False,

            # Cognitive sub-features
            "cog_frozen": True,
            "cog_thinking": True,

            # Forensic sub-features
            "for_suppress": False,
            "for_failure": True,
            "for_ownership_transfers": False,
            "for_fragmentation": False
        }
        
    def toggle(self, layer, state=None):
        if state is None:
            self.layers[layer] = not self.layers[layer]
        else:
            self.layers[layer] = state
            
    def is_enabled(self, layer):
        return self.layers.get(layer, False)


class DebugOverlay:
    """
    Optional on-screen debug overlay for cognitive memory state.

    Draws lightweight text annotations next to bounding boxes and a
    warm memory status panel. All rendering is pure cv2.putText /
    cv2.rectangle — no memory allocations, no tensor ops.

    Args:
        enabled: If False, all draw calls are no-ops.
    """

    def __init__(self, enabled=False):
        self.enabled = enabled
        self._flash_events = []   # Transient resurrection flash events
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_small = cv2.FONT_HERSHEY_PLAIN

    def toggle(self):
        """Toggle debug overlay on/off."""
        self.enabled = not self.enabled
        return self.enabled

    def flash_resurrection(self, track_id, similarity, bbox):
        """
        Queue a transient resurrection flash event.

        Args:
            track_id:   The resurrected track ID.
            similarity: The match similarity score.
            bbox:       [x1, y1, x2, y2] position for the flash.
        """
        self._flash_events.append({
            "track_id": track_id,
            "similarity": similarity,
            "bbox": bbox,
            "birth": time.perf_counter(),
            "ttl": 2.0,  # seconds to display
        })

    def draw_all(self, frame, brain, tracker):
        """
        Master draw call — composes all overlays onto the frame.

        Args:
            frame:   The BGR frame to draw on (mutated in place).
            brain:   CognitiveMemory instance.
            tracker: StrongSORTTracker instance.
        """
        if not self.enabled:
            return

        # Draw per-track debug info
        for track in tracker.tracks:
            if track.is_confirmed and track.time_since_update == 0:
                self._draw_track_info(frame, track, brain)

        # Draw warm memory panel
        self._draw_warm_panel(frame, brain)

        # Draw resurrection flashes
        self._draw_flashes(frame)

    def _draw_track_info(self, frame, track, brain):
        """Draw debug annotations next to a tracked person's bbox."""
        x1, y1, x2, y2 = map(int, track.smooth_box)

        # Get state from brain
        state = brain.get_identity_state(track.id) or "?"
        debug = brain.get_debug_info(track.id)

        # Pick color based on state
        if debug and debug.get("locked"):
            state_clr = _CLR_LOCKED
        elif state == "ACTIVE":
            state_clr = _CLR_ACTIVE
        elif state == "UNCERTAIN":
            state_clr = _CLR_UNCERTAIN
        else:
            state_clr = _CLR_DIM

        # Draw info block to the right of the bounding box
        tx = x2 + 5
        ty = y1

        lines = [
            (f"{state}", state_clr),
        ]

        if debug:
            qual = debug.get("quality", 0)
            lines.append((f"Q:{qual:.2f}", _CLR_DIM))
            if debug.get("locked"):
                reasons = debug.get("lock_reasons", [])
                reason_str = ",".join(reasons) if reasons else "UNCERTAIN"
                # Abbreviate reasons to save space
                reason_str = reason_str.replace("frame_edge", "edge")
                reason_str = reason_str.replace("aspect_shift", "aspect")
                reason_str = reason_str.replace("area_drop", "area")
                reason_str = reason_str.replace("low_confidence", "conf")
                lines.append((f"LOCKED:{reason_str}", _CLR_LOCKED))
            gallery_n = debug.get("gallery_size", 0)
            if gallery_n > 0:
                lines.append((f"G:{gallery_n}", _CLR_DIM))

        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (tx, ty + 14 * (i + 1)),
                        self._font_small, 1.0, color, 1)

    def _draw_warm_panel(self, frame, brain):
        """Draw warm memory status panel in bottom-right corner."""
        if brain.warm_count == 0:
            return

        h, w = frame.shape[:2]
        panel_w = 220
        panel_h = 20 + brain.warm_count * 16
        panel_h = min(panel_h, 200)  # Cap height

        px = w - panel_w - 10
        py = h - panel_h - 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                      _CLR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, f"WARM MEMORY ({brain.warm_count})",
                    (px + 5, py + 14),
                    self._font_small, 1.0, _CLR_WARM, 1)

        # List entries (limited to avoid overflow)
        row = 0
        for tid, mem in list(brain.warm_memory.items())[:10]:
            row += 1
            conf = mem.get("confidence", 0)
            imp = mem.get("importance", 0)
            text = f"ID {tid}: C={conf:.2f} I={imp:.1f}"
            cv2.putText(frame, text,
                        (px + 5, py + 14 + row * 16),
                        self._font_small, 0.9, _CLR_DIM, 1)

    def _draw_flashes(self, frame):
        """Draw transient resurrection flash events."""
        now = time.perf_counter()
        alive = []

        for evt in self._flash_events:
            age = now - evt["birth"]
            if age > evt["ttl"]:
                continue
            alive.append(evt)

            # Fade alpha based on age
            alpha = max(0.0, 1.0 - age / evt["ttl"])
            bbox = evt["bbox"]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int(bbox[1]) - 30

            # Pulsing color
            intensity = int(255 * alpha)
            color = (0, intensity, intensity)

            text = f"RECALLED ID {evt['track_id']} ({evt['similarity']:.2f})"
            cv2.putText(frame, text, (cx - 80, cy),
                        self._font, 0.5, color, 2)

        self._flash_events = alive
