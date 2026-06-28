"""
SELFWATCH — Detection Verification Script

Runs RF-DETR on a video file detecting all 6 theft-relevant classes:
  person(0), backpack(26), handbag(30), suitcase(32), laptop(72), cell_phone(76)

Logs every detection per frame, renders labeled boxes, produces summary.

Usage:
  python verify_detections.py --video <path> [--conf 0.35] [--out <path>]
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch

from detectors.rtdetr_detector import RTDETRDetector

# ── 0-indexed COCO class IDs ────────────────────────────────────────
TARGET_CLASSES = [0, 26, 30, 32, 72, 76]
CLASS_NAMES = {
    0:  "Person",
    26: "Backpack",
    30: "Handbag",
    32: "Suitcase",
    72: "Laptop",
    76: "CellPhone",
}

# Colors per class (BGR for OpenCV)
CLASS_COLORS = {
    0:  (0,   255, 0),    # Person — green
    26: (255, 128, 0),    # Backpack — orange
    30: (255, 0,   255),  # Handbag — magenta
    32: (0,   128, 255),  # Suitcase — gold
    72: (255, 0,   0),    # Laptop — red
    76: (0,   255, 255),  # CellPhone — cyan
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    parser = argparse.ArgumentParser(description="SELFWATCH Detection Verification")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence threshold")
    parser.add_argument("--out", type=str, default=None,
                        help="Output video path (optional)")
    parser.add_argument("--variant", type=str, default="nano",
                        choices=["nano", "small", "base", "medium", "large"],
                        help="RF-DETR variant")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # ── Build detector ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELFWATCH Detection Verification")
    print(f"  Variant    : RF-DETR-{args.variant.capitalize()}")
    print(f"  Conf thresh: {args.conf}")
    print(f"  Classes    : {', '.join(CLASS_NAMES.values())}")
    print(f"  IDs        : {TARGET_CLASSES}")
    print(f"  Video      : {args.video}")
    print(f"{'='*60}\n")

    detector = RTDETRDetector(
        variant=args.variant,
        resolution=384,
        use_amp=True,
        compile_model=False,
    )
    detector.warmup(input_size=(960, 540))

    # ── Open video ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames\n")

    # ── Output video ─────────────────────────────────────────────────
    out_writer = None
    if args.out:
        out_writer = cv2.VideoWriter(
            args.out,
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(fps, 15.0),
            (width, height),
        )

    # ── Stats tracking ───────────────────────────────────────────────
    det_log  = []            # list of dicts per detection
    frame_has_class = defaultdict(set)   # frame_idx -> set of class_ids seen
    per_class = {cid: {"count": 0, "conf_sum": 0.0, "max_conf": 0.0, "frames_seen": set()}
                 for cid in TARGET_CLASSES}

    # ── CSV log ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"detection_log_{timestamp}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "class_id", "class_name", "confidence",
                         "x1", "y1", "x2", "y2"])

    frame_idx = 0
    loop_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ── Detect ───────────────────────────────────────────────
        result = detector.detect(
            frame,
            conf_threshold=args.conf,
            target_classes=TARGET_CLASSES,
        )

        n_det = len(result.boxes)
        frame_has_class[frame_idx] = set()

        # ── Process detections ───────────────────────────────────
        for di in range(n_det):
            box   = result.boxes[di]
            sc    = float(result.scores[di])
            cid   = int(result.class_ids[di])
            cname = result.labels[di] if result.labels else CLASS_NAMES.get(cid, f"class_{cid}")

            x1, y1, x2, y2 = map(int, box)

            # Log
            entry = {
                "frame": frame_idx, "class_id": cid, "class_name": cname,
                "confidence": sc, "box": box,
            }
            det_log.append(entry)

            csv_writer.writerow([frame_idx, cid, cname, f"{sc:.4f}", x1, y1, x2, y2])

            # Stats
            frame_has_class[frame_idx].add(cid)
            per_class[cid]["count"] += 1
            per_class[cid]["conf_sum"] += sc
            per_class[cid]["max_conf"] = max(per_class[cid]["max_conf"], sc)
            per_class[cid]["frames_seen"].add(frame_idx)

            # ── Draw box ─────────────────────────────────────────
            color = CLASS_COLORS.get(cid, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label: class name + confidence
            label = f"{cname} {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 2)
            label_y = max(y1 - 8, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 4),
                          (x1 + tw + 4, label_y + 4), color, -1)
            cv2.putText(frame, label, (x1 + 2, label_y),
                        FONT, 0.5, (0, 0, 0), 2)

        # ── Per-frame status line ────────────────────────────────
        classes_str = ", ".join(
            f"{CLASS_NAMES[cid]}" for cid in sorted(frame_has_class[frame_idx])
        ) if frame_has_class[frame_idx] else "NONE"
        print(f"  F#{frame_idx:>5d}: {n_det} detections [{classes_str}]")

        # ── Write output video ───────────────────────────────────
        if out_writer:
            out_writer.write(frame)

    # ── Cleanup ─────────────────────────────────────────────────
    cap.release()
    if out_writer:
        out_writer.release()
    csv_file.close()

    elapsed = time.perf_counter() - loop_start
    avg_fps = frame_idx / elapsed

    # ── Compute gap frames (frames where a class disappeared) ────
    # A "gap" is any frame where a class was not detected between
    # frames where it WAS detected (i.e., intermittent detection).
    # We also track contiguous runs and report frames where a class
    # was completely absent after having been seen.
    gaps = {}
    for cid in TARGET_CLASSES:
        seen_frames = sorted(per_class[cid]["frames_seen"])
        if len(seen_frames) < 2:
            gaps[cid] = []
            continue

        # Find gaps (frames between first and last where class NOT seen)
        first, last = seen_frames[0], seen_frames[-1]
        seen_set = set(seen_frames)
        gap_frames = [f for f in range(first + 1, last) if f not in seen_set]
        gaps[cid] = gap_frames

    # ── Print summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DETECTION VERIFICATION SUMMARY")
    print(f"  Video: {os.path.basename(args.video)}")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Avg processing FPS: {avg_fps:.1f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  {'Class':<15s} {'Detections':>10s} {'Avg Conf':>9s} {'Max Conf':>9s}  {'Frames':>7s} {'Gap Frames':>11s}")
    print(f"  {'-'*63}")

    for cid in TARGET_CLASSES:
        stats = per_class[cid]
        avg_conf = stats["conf_sum"] / max(1, stats["count"])
        n_gaps = len(gaps[cid])
        n_frames = len(stats["frames_seen"])
        gap_str = f"{n_gaps:>4d}" if n_gaps > 0 else "    —"
        print(f"  {CLASS_NAMES[cid]:<15s} {stats['count']:>10d}  {avg_conf:>7.3f}  {stats['max_conf']:>7.3f}  {n_frames:>5d}  {gap_str:>8s}")

    print(f"{'='*60}")
    print()

    # ── Gap detail for classes with gaps ────────────────────────
    for cid in TARGET_CLASSES:
        if gaps[cid]:
            print(f"  {CLASS_NAMES[cid]} gap frames ({len(gaps[cid])} total):")
            # Show first 10 gap frames
            show = gaps[cid][:10]
            print(f"    Frame indices: {show}")
            if len(gaps[cid]) > 10:
                print(f"    ... and {len(gaps[cid]) - 10} more")

    print(f"\n  CSV log saved to: {csv_path}")
    print(f"  Done.")

    # ── Answers to the four questions ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ANSWERS")
    print(f"{'='*60}")
    for cid in TARGET_CLASSES:
        stats = per_class[cid]
        detected = stats["count"] > 0
        print(f"  {CLASS_NAMES[cid]:<15s}: {'DETECTED' if detected else 'NOT DETECTED'} "
              f"({stats['count']} boxes across {len(stats['frames_seen'])} frames)")


if __name__ == "__main__":
    main()
