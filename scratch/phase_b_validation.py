"""
Phase B Validation Script
Runs the SELFWATCH pipeline on a video (or camera) to measure:
- Average FPS
- Number of active person tracks
- Number of active object tracks
- Identity switch occurrences (approximated or logged)
"""
import sys
import os
import cv2
import time
from collections import defaultdict
import numpy as np

# Ensure project root is in sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from engine.pipeline import SelfWatchPipeline

# Monkey-patch config to ensure we run headless
config.MEMORY_DEBUG_MODE = False

def run_validation(source, num_frames=300):
    print(f"Initializing components...")
    from detectors import RTDETRDetector
    from reid import EmbeddingExtractor
    from trackers import StrongSORTTracker
    
    detector = RTDETRDetector(
        variant=config.DETECTOR_VARIANT,
        resolution=config.DETECTOR_RESOLUTION,
        use_amp=config.DETECTOR_AMP,
        compile_model=config.DETECTOR_COMPILE,
    )
    detector.warmup()
    
    reid = EmbeddingExtractor(
        weights_path=config.REID_WEIGHTS,
        device=detector.get_device(),
        half=config.REID_HALF,
    )
    
    tracker = StrongSORTTracker(
        appearance_weight=config.TRACKER_APPEARANCE_WEIGHT,
        high_thresh=config.TRACKER_HIGH_THRESH,
        low_thresh=config.TRACKER_LOW_THRESH,
        iou_thresh=config.TRACKER_IOU_THRESH,
        max_cosine_dist=config.TRACKER_MAX_COSINE_DIST,
        max_lost=config.TRACKER_MAX_LOST,
        confirm_threshold=config.TRACKER_CONFIRM_THRESHOLD,
        embedding_history=config.TRACKER_EMBEDDING_HISTORY,
        min_quality_score=config.TRACKER_MIN_QUALITY_SCORE,
    )
    
    pipeline = SelfWatchPipeline(
        detector=detector,
        reid=reid,
        tracker=tracker,
        enable_debug_overlay=False
    )
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source {source}")
        return
        
    print(f"Starting validation for {num_frames} frames on {source}...")
    
    frame_times = []
    
    # Metrics
    person_track_counts = []
    object_track_counts = []
    
    # We can inspect the tracks directly from pipeline.tracker.tracks
    # Track ID mapping for basic switch tracking (just tracking how many unique IDs we see vs active tracks)
    seen_person_lids = set()
    seen_object_lids = set()
    
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (960, 540))
        
        t0 = time.perf_counter()
        
        # Run pipeline iteration
        frame, _ = pipeline.process_frame(frame, frame_delta=1, frame_index=frame_count)
        
        t1 = time.perf_counter()
        
        # We only count pipeline time, not video reading/resizing
        frame_times.append(t1 - t0)
        
        # Count active tracks
        active_people = 0
        active_objects = 0
        
        for t in pipeline.tracker.tracks:
            if t.state == 0:  # STATE_ACTIVE = 0, assuming it's confirmed and active
                if t.time_since_update > 0:
                    continue # only count currently matched
                    
                if getattr(t, 'class_id', 0) == 0:
                    active_people += 1
                    seen_person_lids.add(t.local_id)
                else:
                    active_objects += 1
                    seen_object_lids.add(t.local_id)
        
        person_track_counts.append(active_people)
        object_track_counts.append(active_objects)
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {active_people} persons, {active_objects} objects, {(t1-t0)*1000:.1f}ms")
            
        frame_count += 1
        
    cap.release()
    
    avg_ms = np.mean(frame_times) * 1000
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    
    print("\n" + "="*50)
    print("  PHASE B VALIDATION REPORT")
    print("="*50)
    print(f"  Processed Frames:       {frame_count}")
    print(f"  Average FPS:            {avg_fps:.1f} FPS (avg {avg_ms:.1f} ms/frame)")
    print(f"  Avg Active Persons:     {np.mean(person_track_counts):.2f}")
    print(f"  Avg Active Objects:     {np.mean(object_track_counts):.2f}")
    print(f"  Total Unique Person IDs:{len(seen_person_lids)}")
    print(f"  Total Unique Object IDs:{len(seen_object_lids)}")
    print("="*50)

if __name__ == "__main__":
    # Test on camera 0 for a short burst
    run_validation(0, num_frames=100)
