import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import cv2
from detectors.trt_detector import TRTDetector
from engine.pipeline import SelfWatchPipeline
import config

video_path = "C:/Users/Dev/Downloads/sh1.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

print("=== FORENSIC INVESTIGATION: BAG DISAPPEARANCE IN ACTUAL PIPELINE ===")
detector = TRTDetector(resolution=384, fallback=True)

# Stage 1: Raw Detector Output (No Class Filter)
raw_res = detector.detect(frame, conf_threshold=0.25, target_classes=None)
print("\n[STAGE 1] Raw Detector Detections (target_classes=None):")
for b, sc, cid, lbl in zip(raw_res.boxes, raw_res.scores, raw_res.class_ids, raw_res.labels):
    print(f"  Box={b.tolist()} Conf={sc:.3f} ClassID={cid} Label={lbl}")

# Stage 2: Detector Output inside MultiCameraPipeline (target_classes=[0])
filt_res = detector.detect(frame, conf_threshold=0.35, target_classes=[0])
print("\n[STAGE 2] Detector Output with target_classes=[0] (MultiCameraPipeline default):")
for b, sc, cid, lbl in zip(filt_res.boxes, filt_res.scores, filt_res.class_ids, filt_res.labels):
    print(f"  Box={b.tolist()} Conf={sc:.3f} ClassID={cid} Label={lbl}")

# Stage 3: Inside SelfWatchPipeline.process_frame_with_precomputed
# Check pipeline geometry filter (by2 - by1 >= 60 and bx2 - bx1 >= 25)
print("\n[STAGE 3] SelfWatchPipeline Geometry Filtering:")
for b, sc, cid, lbl in zip(raw_res.boxes, raw_res.scores, raw_res.class_ids, raw_res.labels):
    bx1, by1, bx2, by2 = b
    h, w = by2 - by1, bx2 - bx1
    passed = (h >= 60 and w >= 25)
    print(f"  Label={lbl} h={h:.1f} w={w:.1f} -> Passed Geometry Filter: {passed}")

print("\n=== CONCLUSION: EXACT STAGES WHERE BAGS DISAPPEAR ===")
print("1. MultiCameraPipeline.step() calls detect_batch(..., target_classes=[0]). Handbags/Backpacks (class 24, 26) are discarded here.")
print("2. SelfWatchPipeline.process_frame_with_precomputed calls detect(..., target_classes=[PERSON_CLASS=0]).")
print("3. Even if allowed, SelfWatchPipeline discards class_ids before calling tracker.update().")
