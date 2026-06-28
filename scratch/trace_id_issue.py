import cv2
from multicam.multicam_pipeline import MultiCameraPipeline

video_path = "C:/Users/Dev/Downloads/sh1.mp4"
pipeline = MultiCameraPipeline(detector_variant="nano", detector_backend="tensorrt")
pipeline.add_camera(video_path, label="RealVideo")

cam = pipeline.cameras[0]
cam.open()

out_lines = []

for frame_idx in range(1, 440):
    ret, frame = cam.read_frame(timeout=0.2)
    if not ret or frame is None:
        continue

    res, stats = cam.process_frame(frame, cross_camera_matcher=pipeline.cross_camera_matcher)

    if frame_idx in (1, 100, 200, 290, 295, 296, 297, 300, 350, 393, 394, 395, 396, 397, 398, 399, 400, 435, 436):
        gids = stats.get("multicam_active_gids", {})
        out_lines.append(f"\n[FRAME {frame_idx}] Active local->global: {gids}")
        for t in cam.pipeline.tracker.tracks:
            st = {0:"TENT", 1:"CONF", 2:"LOST"}.get(t.state, str(t.state))
            out_lines.append(f"  local={t.local_id} gid={t.global_id} state={st} hits={t.total_hits} tsu={t.time_since_update} box={t.smooth_box.astype(int).tolist()}")

cam.close()

with open("scratch/id_trace_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print("TRACE SAVED")
