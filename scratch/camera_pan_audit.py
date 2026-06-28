import sys
import os
import cv2
import time
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from engine.pipeline import SelfWatchPipeline

# Ensure headless mode for testing
config.MEMORY_DEBUG_MODE = False

def run_camera_pan_audit(source_path):
    print(f"Initializing audit on source: {source_path}")
    from detectors import RTDETRDetector
    from reid import EmbeddingExtractor
    from trackers import StrongSORTTracker
    
    detector = RTDETRDetector(
        variant=config.DETECTOR_VARIANT,
        resolution=config.DETECTOR_RESOLUTION,
        use_amp=config.DETECTOR_AMP,
        compile_model=config.DETECTOR_COMPILE,
    )
    
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
    
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {source_path}")
        return
        
    frame_count = 0
    print("Beginning forensic audit...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (960, 540))
        pipeline.process_frame(frame, frame_delta=1, frame_index=frame_count)
        frame_count += 1
        
    cap.release()
    print("Audit complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scratch/camera_pan_audit.py <path_to_video>")
    else:
        run_camera_pan_audit(sys.argv[1])
