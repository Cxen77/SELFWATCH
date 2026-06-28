import sys
import os
import cv2
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath('.'))

from detectors.rtdetr_detector import RTDETRDetector

video_path = "C:/Users/Dev/Downloads/sh3.mp4"
if not os.path.exists(video_path):
    video_path = "C:/Users/Dev/Downloads/vidssave.com HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store 720p.mp4"

print(f"Auditing video: {video_path}")
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total video frames: {total_frames}")

detector = RTDETRDetector(variant="nano", resolution=384)
detector.warmup()

class_histogram = Counter()
bottle_detections = []
all_region_classes = Counter()

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    # detect() will trigger our [RAW DET] log!
    res = detector.detect(frame, conf_threshold=0.05, target_classes=None)
    
    for cid, cname, score, b in zip(res.class_ids, res.labels, res.scores, res.boxes):
        class_histogram[(cid, cname)] += 1
        all_region_classes[(cid, cname)] += 1
        
        if cid == 43: # bottle
            bx = f"({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})"
            bottle_detections.append((frame_idx, score, bx))

print("\n" + "="*50)
print("HISTOGRAM OF DETECTIONS (>0.05 conf)")
print("="*50)
print(f"{'Class ID':<10} | {'Class Name':<20} | {'Count':<10}")
print("-" * 45)
for (cid, cname), count in class_histogram.most_common():
    print(f"{cid:<10} | {cname:<20} | {count:<10}")

print("\n" + "="*50)
print("SPECIFIC BOTTLE (class 43) ANALYSIS")
print("="*50)
if bottle_detections:
    print(f"Class 43 (bottle) detected {len(bottle_detections)} times!")
    confs = [s for _, s, _ in bottle_detections]
    frames = [f for f, _, _ in bottle_detections]
    print(f"Confidence range: {min(confs):.3f} - {max(confs):.3f}")
    print(f"Frames: {min(frames)} to {max(frames)}")
else:
    print("Class 43 (bottle) NEVER detected!")
    print("Top 10 classes detected across all regions:")
    for (cid, cname), count in all_region_classes.most_common(10):
        print(f"  {cid} ({cname}): {count}")
