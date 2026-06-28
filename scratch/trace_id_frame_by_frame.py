import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import cv2
import time
from multicam.multicam_pipeline import MultiCameraPipeline

video_path = "C:/Users/Dev/Downloads/sh1.mp4"
pipeline = MultiCameraPipeline(detector_variant="nano", detector_backend="tensorrt")
pipeline.add_camera(video_path, label="RealVideo")

cam = pipeline.cameras[0]
cam.open()

print("=== FRAME-BY-FRAME TRACE OF REAL VIDEO (sh1.mp4) AROUND OCCLUSION ===")

with open("scratch/occlusion_trace.txt", "w", encoding="utf-8") as out_f:
    for frame_idx in range(1, 445):
        ret, frame = cam.read_frame(timeout=0.5)
        if not ret or frame is None:
            if cam._stream_ended:
                print(f"End of stream at frame {frame_idx}")
                break
            continue

        # Process frame
        res, stats = cam.process_frame(frame, cross_camera_matcher=pipeline.cross_camera_matcher)

        if 380 <= frame_idx <= 440:
            gids = stats.get("multicam_active_gids", {})
            out_f.write(f"\n--- [FRAME {frame_idx}] Active Tracks (local->global): {gids} ---\n")
            trk_obj = cam.pipeline.tracker
            for t in trk_obj.tracks:
                state_str = {0:"TENT", 1:"CONF", 2:"LOST"}.get(t.state, str(t.state))
                out_f.write(f"  Track local={t.local_id} gid={t.global_id} state={state_str} hits={t.total_hits} time_lost={t.time_since_update} box={t.smooth_box.astype(int).tolist()} method={getattr(t, 'last_assoc_method', 'N/A')}\n")

cam.close()
print("=== TRACE COMPLETE ===")
