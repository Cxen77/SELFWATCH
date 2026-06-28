import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import cv2
import time
from multicam.multicam_pipeline import MultiCameraPipeline

print("=== STARTING ACTUAL APPLICATION TRACE ON C:/Users/Dev/Downloads/sh1.mp4 ===")
pipeline = MultiCameraPipeline(detector_variant="nano", detector_backend="tensorrt")
pipeline.add_camera("C:/Users/Dev/Downloads/sh1.mp4", label="RealVideo")

# Open camera
cam = pipeline.cameras[0]
cam.open()

# We want to intercept cam.process_frame or pipeline step to log:
# - detector result
# - tracker input
# - renderer input
# - association / birth / lost / recovery

for frame_idx in range(1, 450):
    ret, frame = cam.read_frame(timeout=0.2)
    if not ret or frame is None:
        if cam._stream_ended:
            print(f"End of video reached at frame {frame_idx}")
            break
        continue

    # Let's see what detector produces inside process_frame
    # In camera_stream.py line 253: process_frame_with_precomputed or process_frame
    res, stats = cam.process_frame(frame, cross_camera_matcher=pipeline.cross_camera_matcher)
    
    # Check active tracks
    active = stats.get("active_dict", {})
    gids = stats.get("multicam_active_gids", {})
    if frame_idx % 30 == 0 or frame_idx > 380:
        print(f"[FRAME {frame_idx}] Active local->global: {gids}")

print("=== TRACE COMPLETE ===")
cam.close()
