"""Verify the class ID remapping fix works end-to-end."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cv2
import numpy as np
from detectors import RFDETRDetector

print("Loading RF-DETR Base...")
det = RFDETRDetector(variant="base", resolution=560, compile_model=False)

# Grab a camera frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: no camera frame")
    sys.exit(1)

print(f"\nFrame: {frame.shape}")

# Test WITHOUT class filter
result_all = det.detect(frame, conf_threshold=0.3)
print(f"\n[No filter] Detections: {result_all.count}")
for i in range(min(5, result_all.count)):
    print(f"  class_id={result_all.class_ids[i]}, "
          f"label={result_all.labels[i]}, "
          f"conf={result_all.scores[i]:.3f}, "
          f"box={result_all.boxes[i]}")

# Test WITH person filter (class_id=0 in 0-indexed)
result_person = det.detect(frame, conf_threshold=0.3, target_classes=[0])
print(f"\n[Person only, class=0] Detections: {result_person.count}")
for i in range(min(5, result_person.count)):
    print(f"  class_id={result_person.class_ids[i]}, "
          f"label={result_person.labels[i]}, "
          f"conf={result_person.scores[i]:.3f}, "
          f"box={result_person.boxes[i]}")

if result_person.count > 0:
    print("\n✅ FIX VERIFIED — person detections are flowing through!")
else:
    print("\n⚠️ No person detected (might not be visible in frame)")
    if result_all.count > 0:
        print("   But other objects were detected, so the detector works.")
