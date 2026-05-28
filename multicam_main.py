"""
SELFWATCH -- Multi-Camera Persistent Person Tracking (Phase 1)

Entry point for running 2+ camera streams with shared global identity space.

Pipeline:
    Camera0 → RT-DETR → StrongSORT → Local IDs ──┐
    Camera1 → RT-DETR → StrongSORT → Local IDs ──┤
    ...                                           │
                                                  ▼
                                    Shared Global Identity Space
                                    Cross-Camera ReID Matching
                                    Dormant Identity Memory
                                    Entry/Exit Event Tracking

Usage:
    python multicam_main.py                       # Two webcams (0, 1)
    python multicam_main.py --sources 0 1 2       # Three webcams
    python multicam_main.py --sources cam1.mp4 cam2.mp4  # Two video files
    python multicam_main.py --sources 0 cam2.mp4  # Mixed sources
"""

import argparse
import torch

import config
from multicam import MultiCameraPipeline


def main():
    parser = argparse.ArgumentParser(
        description="SELFWATCH Multi-Camera — Phase 1")

    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Video sources: integers for webcams, strings for files/RTSP. "
             "Default: 0 1 (two webcams)")
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Human-readable labels for each camera. "
             "Default: Camera0, Camera1, ...")
    parser.add_argument(
        "--variant", type=str, default=config.DETECTOR_VARIANT,
        choices=["nano", "medium", "large"],
        help="RT-DETR variant")
    parser.add_argument(
        "--resolution", type=int, default=config.DETECTOR_RESOLUTION,
        help="Detector input resolution")
    parser.add_argument(
        "--fp16", action="store_true", default=config.DETECTOR_AMP,
        help="Enable FP16 inference")
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable FP16 inference")
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.70,
        help="Cross-camera ReID similarity threshold (0-1)")
    parser.add_argument(
        "--max-dormant-time", type=float, default=300.0,
        help="Max seconds for dormant identity survival")
    parser.add_argument(
        "--max-fps", type=float, default=15.0,
        help="Target FPS limit")
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Show each camera in a separate window instead of grid")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug overlay")

    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    # ── Interactive model selection ──────────────────────────────────
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

    # ── Parse sources ────────────────────────────────────────────────
    if args.sources is None:
        sources = [0, 1]  # Default: two webcams
    else:
        sources = []
        for s in args.sources:
            try:
                sources.append(int(s))
            except ValueError:
                sources.append(s)  # Keep as string (file/RTSP)

    labels = args.labels or [f"Camera{i}" for i in range(len(sources))]
    if len(labels) < len(sources):
        labels.extend(
            [f"Camera{i}" for i in range(len(labels), len(sources))])

    # ── Print configuration ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELFWATCH Multi-Camera — Phase 1")
    print(f"{'='*60}")
    print(f"  Cameras: {len(sources)}")
    for i, (src, lbl) in enumerate(zip(sources, labels)):
        print(f"    [{i}] {lbl}: source={src}")
    print(f"  Detector: RT-DETR {args.variant.upper()} @ {args.resolution}")
    print(f"  FP16: {args.fp16}")
    print(f"  Cross-cam threshold: {args.similarity_threshold}")
    print(f"  Max dormant time: {args.max_dormant_time}s")
    print(f"  CUDA: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if torch.cuda.is_available() else ""))
    print(f"{'='*60}\n")

    # ── Build multi-camera pipeline ──────────────────────────────────
    pipeline = MultiCameraPipeline(
        detector_variant=args.variant,
        detector_resolution=args.resolution,
        use_fp16=args.fp16,
        similarity_threshold=args.similarity_threshold,
        max_dormant_time=args.max_dormant_time,
        enable_debug=args.debug,
    )

    # Add cameras
    for i, (src, lbl) in enumerate(zip(sources, labels)):
        pipeline.add_camera(source=src, label=lbl, camera_id=i)

    # ── Run ──────────────────────────────────────────────────────────
    pipeline.run(
        max_fps=args.max_fps,
        display=True,
        grid_layout=not args.no_grid,
    )


if __name__ == "__main__":
    main()
