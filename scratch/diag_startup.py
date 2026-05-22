"""
Diagnostic: trace the exact box positions through the pipeline at startup.
Run this to see what the pipeline actually produces on the first few frames.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time

# Find the video
video_path = None
for root, dirs, files in os.walk(r"c:\Users\Dev\Desktop\SELFWATCH"):
    for f in files:
        if f.endswith(('.mp4', '.avi', '.mkv')) and 'venv' not in root:
            video_path = os.path.join(root, f)
            break
    if video_path:
        break

if not video_path:
    print("No video file found! Please specify path.")
    sys.exit(1)

print(f"Using video: {video_path}")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# Init pipeline
from engine.pipeline import SelfWatchPipeline
from detectors.rf_detr_detector import RFDETRDetector
from reid.osnet_reid import OSNetReID
from trackers.strongsort_tracker import StrongSORTTracker
import config

print("Loading detector...")
det = RFDETRDetector()
print("Loading reid...")
reid = OSNetReID()
print("Loading tracker...")
tracker = StrongSORTTracker(
    max_age=config.TRACKER_MAX_AGE,
    confirm_threshold=config.TRACKER_CONFIRM_THRESHOLD,
)
pipeline = SelfWatchPipeline(det, reid, tracker, enable_debug_overlay=False)
print("Pipeline ready.\n")

# Process first 10 frames and log everything
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    t0 = time.perf_counter()
    annotated, stats = pipeline.process_frame(frame)
    t1 = time.perf_counter()
    
    track_states = stats.get("track_states", {})
    active = stats.get("active_dict", {})
    raw_dets = stats.get("raw_detections", 0)
    active_tracks = stats.get("active_tracks", 0)
    
    print(f"Frame {i+1}: {(t1-t0)*1000:.0f}ms | "
          f"raw_dets={raw_dets} | "
          f"active_tracks={active_tracks} | "
          f"track_states_for_UI={len(track_states)}")
    
    # Show which tracks are confirmed vs tentative
    for track in tracker.tracks:
        state_str = "CONFIRMED" if track.is_confirmed else "TENTATIVE" if track.state == 0 else "LOST"
        print(f"  Track {track.id}: {state_str} | "
              f"hits={track.total_hits} | "
              f"age={track.age} | "
              f"tsu={track.time_since_update} | "
              f"box={[int(x) for x in track.smooth_box]}")
    
    if track_states:
        print(f"  → UI will show: {list(track_states.keys())}")
    else:
        print(f"  → UI will show: NOTHING (no confirmed tracks with tsu<=5)")
    print()

cap.release()
pipeline.close()
