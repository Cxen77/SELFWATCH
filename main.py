"""
SELFWATCH -- Single-Camera Persistent Person Tracking

Pipeline: RT-DETR -> StrongSORT -> OSNet Embeddings -> Stable IDs
"""

import argparse
import cv2
import time
import numpy as np

import config
from detectors import RTDETRDetector, BaseDetector
from reid import EmbeddingExtractor
from trackers import StrongSORTTracker
from memory.cognitive import CognitiveMemory
from memory.event_log import CognitiveEventLogger
from memory.metrics import TrackingMetrics
from memory.debug_overlay import DebugOverlay
from memory.phantom import PhantomTracker
from memory.contradiction import ContradictionDetector
from memory.attention import CognitiveAttention
from memory.topology import SceneTopology
from memory.gait import GaitSignature
from engine.pipeline import SelfWatchPipeline


# ==========================================
# DETECTOR FACTORY
# ==========================================
PERSON_CLASS = 0


def build_detector(args) -> BaseDetector:
    return RTDETRDetector(
        variant=args.variant,
        resolution=args.resolution,
        use_amp=args.fp16,
        compile_model=not args.no_compile,
        pretrain_weights=args.rtdetr_weights,
    )


def id_color(tid):
    rng = np.random.RandomState(tid * 7)
    return tuple(int(c) for c in rng.randint(100, 255, 3))


# ==========================================
# MAIN LOOP
# ==========================================
def main():
    import torch

    parser = argparse.ArgumentParser(description="SELFWATCH -- Person Tracking")
    parser.add_argument("--variant", type=str, default=config.DETECTOR_VARIANT,
                        choices=["nano", "medium", "large"])
    parser.add_argument("--resolution", type=int, default=config.DETECTOR_RESOLUTION)
    parser.add_argument("--rtdetr-weights", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true",
                        default=not config.DETECTOR_COMPILE)
    parser.add_argument("--fp16", action="store_true", default=config.DETECTOR_AMP)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--camera", type=int, default=config.DEFAULT_CAMERA)
    parser.add_argument("--conf", type=float, default=config.DETECTOR_CONF_THRESH)
    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    # ── Phase 1 Profiling Overrides ──────────────────────────────────
    print("\n--- Model Selection ---")
    print("1 = RT-DETR-Large")
    print("2 = RT-DETR-Medium")
    print("3 = RT-DETR-Nano")
    choice = input("Select variant (1/2/3) [default=3]: ").strip()
    
    if choice == "1":
        args.variant = "large"
        args.resolution = 704
    elif choice == "2":
        args.variant = "medium"
        args.resolution = 512
    else:
        args.variant = "nano"
        args.resolution = 384

    args.fp16 = True
    args.no_compile = True
    max_fps = 15.0
    frame_time_target = 1.0 / max_fps

    # ── Build detector ───────────────────────────────────────────────
    detector = build_detector(args)

    # ── Build OSNet ReID extractor ───────────────────────────────────
    reid = EmbeddingExtractor(
        weights_path=config.REID_WEIGHTS,
        device=detector.get_device(),
        half=config.REID_HALF,
    )

    # ── Build StrongSORT tracker ─────────────────────────────────────
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

    # ── Open camera ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)

    print(f"\n{'='*55}")
    print(f"[SELFWATCH] Detector   : {detector.get_name()}")
    print(f"[SELFWATCH] Tracker    : StrongSORT "
          f"(app={config.TRACKER_APPEARANCE_WEIGHT:.0%})")
    print(f"[SELFWATCH] ReID       : OSNet x1.0 (MSMT17, 512-dim)")
    print(f"[SELFWATCH] Device     : {detector.get_device()}")
    print(f"[SELFWATCH] CUDA       : {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if torch.cuda.is_available() else ""))
    print(f"[SELFWATCH] Filter     : Person only (class {PERSON_CLASS})")
    print(f"[SELFWATCH] Max lost   : {config.TRACKER_MAX_LOST} frames")
    print(f"[SELFWATCH] Confirm    : {config.TRACKER_CONFIRM_THRESHOLD} hits")
    print(f"{'='*55}")
    print(f"[SELFWATCH] Press 'q' to quit.\n")

    detector.warmup()
    
    pipeline = SelfWatchPipeline(
        detector=detector,
        reid=reid,
        tracker=tracker,
        enable_debug_overlay=config.MEMORY_DEBUG_MODE
    )

    prev_time = time.time()
    raw_frame_index = 0
    
    while True:
        loop_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            break
        raw_frame_index += 1

        frame, stats = pipeline.process_frame(
            frame, frame_delta=1, frame_index=raw_frame_index)

        # ── HUD ──────────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        topo_str = "READY" if (pipeline.topology and pipeline.topology.is_ready) else "learning"
        cv2.putText(frame,
                    f"StrongSORT | RT-DETR {args.variant.upper()} | "
                    f"OSNet | Topo:{topo_str}",
                    (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        # ── Profiling Metrics Print ──────────────────────────────────
        total_fps_actual = 1.0 / (time.perf_counter() - loop_start + 1e-6)
        print(f"\r[Profiling] Det: {stats['det_ms']:4.1f}ms | ReID: {stats['reid_ms']:4.1f}ms | Trk: {stats['trk_ms']:4.1f}ms | Total FPS: {total_fps_actual:4.1f} ", end="")

        cv2.imshow("SELFWATCH - Person Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            state = pipeline.debug_overlay.toggle()
            print(f"\n[DEBUG] Overlay {'ON' if state else 'OFF'}")
        # ── FPS Limiter ──────────────────────────────────────────────
        elapsed = time.perf_counter() - loop_start
        if elapsed < frame_time_target:
            time.sleep(frame_time_target - elapsed)

    print("\n[SELFWATCH] Shutting down...")
    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
