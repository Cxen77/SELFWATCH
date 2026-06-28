"""
Shelf Change Detector — category-agnostic object removal via reference frame subtraction.

Detects when ANY object is removed from a shelf zone, regardless of semantic class.
No neural network required. Runs entirely on CPU in ~1-2ms.

Pipeline:
  1. Maintain grayscale reference frame per shelf zone (captured when no person present)
  2. Each frame: diff current vs reference within zone boundaries
  3. Mask out person and known-object bounding boxes
  4. Morphological cleanup (open + close)
  5. Connected component extraction
  6. Emit detections for significant changed regions

Reference lifecycle:
  - Built after STABLE_FRAMES consecutive frames with no person in zone
  - Updated via exponential moving average to handle lighting drift
  - Never updated while a person is present in the zone

Forensic logging: [SHELF_CHANGE] on every detection and reference capture.
"""

import cv2
import numpy as np

from .unknown_object import UNKNOWN_OBJECT


class ShelfChangeDetector:

    def __init__(self, zones, threshold=30, min_area=400,
                 stable_frames=30, confirm_frames=5, person_pad=20):
        self._zones = zones
        self._threshold = threshold
        self._min_area = min_area
        self._stable_frames = stable_frames
        self._confirm_frames = confirm_frames
        self._person_pad = person_pad

        self._references = {}
        self._stable_counts = {}
        self._pending_changes = {}
        self._emitted = {}
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame, person_boxes, known_object_boxes, frame_count):
        if frame is None:
            return []

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        detections = []

        for zone in self._zones:
            zone_name, zx1, zy1, zx2, zy2 = zone

            has_person = self._person_in_zone(person_boxes, zx1, zy1, zx2, zy2)

            if not has_person:
                self._stable_counts[zone_name] = self._stable_counts.get(zone_name, 0) + 1
                if self._stable_counts[zone_name] >= self._stable_frames:
                    self._capture_reference(gray, zone_name, zx1, zy1, zx2, zy2, frame_count)
            else:
                self._stable_counts[zone_name] = 0

            ref = self._references.get(zone_name)
            if ref is None:
                continue

            zone_dets = self._detect_changes(
                gray, ref, zone_name, zx1, zy1, zx2, zy2,
                person_boxes, known_object_boxes, frame_count)
            detections.extend(zone_dets)

        return detections

    def _person_in_zone(self, person_boxes, zx1, zy1, zx2, zy2):
        for pb in person_boxes:
            px1, py1, px2, py2 = pb[:4]
            if px1 < zx2 and px2 > zx1 and py1 < zy2 and py2 > zy1:
                return True
        return False

    def _capture_reference(self, gray, zone_name, zx1, zy1, zx2, zy2, frame_count):
        crop = gray[zy1:zy2, zx1:zx2].copy()
        if zone_name in self._references:
            self._references[zone_name] = cv2.addWeighted(
                self._references[zone_name], 0.95, crop.astype(np.float32), 0.05, 0)
        else:
            self._references[zone_name] = crop.astype(np.float32)
            print(f"[SHELF_CHANGE] zone={zone_name} reference_captured frame={frame_count}")

    def _detect_changes(self, gray, ref, zone_name, zx1, zy1, zx2, zy2,
                        person_boxes, known_object_boxes, frame_count):
        current = gray[zy1:zy2, zx1:zx2]
        ref_u8 = ref.astype(np.uint8)

        diff = cv2.absdiff(current, ref_u8)
        _, mask = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        pad = self._person_pad
        for pb in person_boxes:
            px1 = max(0, int(pb[0]) - zx1 - pad)
            py1 = max(0, int(pb[1]) - zy1 - pad)
            px2 = min(zx2 - zx1, int(pb[2]) - zx1 + pad)
            py2 = min(zy2 - zy1, int(pb[3]) - zy1 + pad)
            if px2 > px1 and py2 > py1:
                mask[py1:py2, px1:px2] = 0

        for ob in known_object_boxes:
            ox1 = max(0, int(ob[0]) - zx1)
            oy1 = max(0, int(ob[1]) - zy1)
            ox2 = min(zx2 - zx1, int(ob[2]) - zx1)
            oy2 = min(zy2 - zy1, int(ob[3]) - zy1)
            if ox2 > ox1 and oy2 > oy1:
                mask[oy1:oy2, ox1:ox2] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            box = [zx1 + cx, zy1 + cy, zx1 + cx + cw, zy1 + cy + ch]
            mean_diff = float(np.mean(diff[cy:cy+ch, cx:cx+cw]))
            confidence = min(0.90, 0.30 + (mean_diff / 100.0) + (area / 5000.0))
            raw_candidates.append({
                "box": box,
                "confidence": round(confidence, 2),
                "class_id": UNKNOWN_OBJECT,
                "zone": zone_name,
                "label": "unknown",
                "area": area,
            })

        confirmed = []
        current_keys = set()

        for cand in raw_candidates:
            key = self._region_key(cand["box"], zone_name)
            current_keys.add(key)

            if key in self._emitted:
                confirmed.append(cand)
                continue

            count = self._pending_changes.get(key, 0) + 1
            self._pending_changes[key] = count

            if count >= self._confirm_frames:
                self._emitted[key] = True
                self._pending_changes.pop(key, None)
                confirmed.append(cand)
                print(f"[SHELF_CHANGE] zone={zone_name} box=({cand['box'][0]},{cand['box'][1]},"
                      f"{cand['box'][2]},{cand['box'][3]}) area={cand['area']} "
                      f"confidence={cand['confidence']} frame={frame_count}")

        stale_pending = [k for k in self._pending_changes if k not in current_keys]
        for k in stale_pending:
            del self._pending_changes[k]

        stale_emitted = [k for k in self._emitted if k not in current_keys]
        for k in stale_emitted:
            del self._emitted[k]
            zone_from_key = k.split(":")[0]
            print(f"[SHELF_CHANGE] zone={zone_from_key} region_returned frame={frame_count}")

        return confirmed

    def _region_key(self, box, zone_name):
        gx = int((box[0] + box[2]) / 2) // 30
        gy = int((box[1] + box[3]) / 2) // 30
        return f"{zone_name}:{gx}:{gy}"

    def has_reference(self, zone_name):
        return zone_name in self._references

    @property
    def zones(self):
        return self._zones

    @property
    def zone_status(self):
        return {z[0]: z[0] in self._references for z in self._zones}
