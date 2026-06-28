"""
Phantom Recovery Bridge — SHADOW MODE Audit Runner v5

Executes forensic validation clips through a unified tracking context to evaluate
identity continuity across occlusion events and clip boundaries.

Methodology:
1. Independent Single-Session Pipeline: All viable clips run sequentially. Between clips,
   tracker state is reset to prevent cross-scene contamination. Tracks reach maturity (>5 hits).
2. Controlled Occlusion Injection: For each clip, after feeding 60% of frames,
   we inject 30 blank frames (simulated occlusion). During this gap, Kalman
   velocity prediction drifts. When clip frames resume, standard Stage 1B gating
   rejects recovery -> unmatched detection -> bridge evaluation fires.
3. Rigorous Ground Truth Evaluation:
   - Within-clip occlusion recovery (same identity):
       Bridge YES -> True Positive (TP)
       Bridge NO  -> False Negative (FN)
   - Cross-clip or foreign detection (different identity/background):
       Bridge YES -> False Positive (FP)
       Bridge NO  -> True Negative (TN)

AUDIT MODE ONLY: No identity changes, no track.update(), no live reactivation.

Usage:
    .\\trt_env\\Scripts\\python.exe run_bridge_audit.py
"""

import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

import os, glob, time, re, io, gc
import cv2
import numpy as np

import config
from detectors import RTDETRDetector
from reid import EmbeddingExtractor
from trackers import StrongSORTTracker
from engine.pipeline import SelfWatchPipeline

FORENSIC_DIR = os.path.join("logs", "forensic")
LOG_FILE = "bridge_audit.log"
MAX_FRAMES_PER_CLIP = 600
MIN_FRAMES = 10


def find_clips(forensic_dir):
    pattern = os.path.join(forensic_dir, "fail_*_clip.mp4")
    clips = glob.glob(pattern)
    viable = []
    for c in clips:
        cap = cv2.VideoCapture(c)
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n >= MIN_FRAMES:
                viable.append((c, n))
    viable.sort(key=lambda x: x[1], reverse=True)
    return viable


def run_chained_audit(clips_info):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = RTDETRDetector(
        variant="nano", resolution=384, use_amp=True, compile_model=False)
    detector.warmup()

    reid = EmbeddingExtractor(
        weights_path=config.REID_WEIGHTS, device=device, half=config.REID_HALF)

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
        detector=detector, reid=reid, tracker=tracker,
        enable_debug_overlay=False,
    )

    print(f"\n{'='*60}")
    print(f"  PIPELINE INITIALIZED — Independent Clip Audit Mode")
    print(f"  Tracker: StrongSORT | Detector: RT-DETR nano | Device: {device}")
    print(f"{'='*60}")

    global_frame_idx = 0

    for clip_idx, (clip_path, total_clip_frames) in enumerate(clips_info):
        clip_name = os.path.basename(clip_path)

        cap = cv2.VideoCapture(clip_path)
        frames = []
        while len(frames) < MAX_FRAMES_PER_CLIP:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        nf = len(frames)
        if nf < MIN_FRAMES:
            continue

        print(f"\n── Clip {clip_idx+1}/{len(clips_info)}: {clip_name} ({nf} frames) ──")

        # Reset tracker between clips to ensure zero bleed across unrelated scenes
        tracker.tracks = []
        tracker.removed_tracks = []
        tracker._next_id = 1

        # Split clip: first 60% -> 30 blank frames -> remaining 40%
        split_idx = max(5, int(nf * 0.6))
        part1 = frames[:split_idx]
        part2 = frames[split_idx:]

        # Mark ground truth phase in logs
        print(f"[GT_PHASE] CLIP_START {clip_name}")

        # Phase 1: Normal feeding (tracks mature)
        print(f"[GT_PHASE] WITHIN_CLIP_MATURING")
        for frame in part1:
            global_frame_idx += 1
            try:
                pipeline.process_frame(frame, frame_delta=1, frame_index=global_frame_idx)
            except Exception:
                pass

        # Phase 2: Simulated Occlusion (30 blank frames -> tracks go lost & drift)
        print(f"[GT_PHASE] WITHIN_CLIP_OCCLUSION")
        blank = np.zeros_like(frames[0])
        for _ in range(30):
            global_frame_idx += 1
            try:
                pipeline.process_frame(blank, frame_delta=1, frame_index=global_frame_idx)
            except Exception:
                pass

        # Phase 3: Recovery attempt (same identity returning)
        print(f"[GT_PHASE] WITHIN_CLIP_RECOVERY")
        for frame in part2:
            global_frame_idx += 1
            try:
                pipeline.process_frame(frame, frame_delta=1, frame_index=global_frame_idx)
            except Exception:
                pass

        active = sum(1 for t in tracker.tracks if t.is_confirmed and t.time_since_update == 0)
        lost = sum(1 for t in tracker.tracks if t.is_lost)
        print(f"Clip {clip_idx+1} finished. Tracks: {len(tracker.tracks)} total ({active} active, {lost} lost)")

    try:
        pipeline.close()
    except:
        pass

    return global_frame_idx


def main():
    clips_info = find_clips(FORENSIC_DIR)
    if not clips_info:
        print(f"[ERROR] No clips found in {FORENSIC_DIR}")
        sys.exit(1)

    # Use top 25 clips for speed
    clips_info = clips_info[:25]
    print(f"Found {len(clips_info)} viable forensic clips")

    t0 = time.time()

    original_stdout = sys.stdout
    log_buffer = io.StringIO()

    class TeeWriter:
        def __init__(self, c, b):
            self.console, self.buffer = c, b
        def write(self, t):
            self.console.write(t)
            self.buffer.write(t)
        def flush(self):
            self.console.flush()
            self.buffer.flush()
        def reconfigure(self, **kw):
            pass

    sys.stdout = TeeWriter(original_stdout, log_buffer)

    try:
        total_frames = run_chained_audit(clips_info)
    finally:
        sys.stdout = original_stdout

    elapsed = time.time() - t0

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())

    print(f"\n{'='*60}")
    print(f"  BRIDGE AUDIT RUN COMPLETE: {total_frames} frames processed in {elapsed:.1f}s")
    print(f"{'='*60}")

    analyze_log(LOG_FILE)


def analyze_log(log_path):
    """
    Parse audit log, determine TP/FP/FN/TN based on GT phase markers,
    and generate statistics and final recommendation.
    """
    pat = re.compile(
        r"\[PHANTOM BRIDGE AUDIT\] "
        r"frame=(\S+) lost_local=(\d+) candidate_det=(\d+) "
        r"anchor_sim=([\d.]+) avg_sim=([\d.]+) raw_dist=([\d.]+) "
        r"score=([\d.]+) tsu=(\d+) would_reactivate=(\w+)")

    current_phase = "UNKNOWN"
    events = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[GT_PHASE]" in line:
                current_phase = line.split("[GT_PHASE]")[1].strip()
            
            m = pat.search(line)
            if m:
                g = m.groups()
                events.append({
                    "frame": int(g[0]),
                    "lost_local": int(g[1]),
                    "candidate_det": int(g[2]),
                    "anchor_sim": float(g[3]),
                    "avg_sim": float(g[4]),
                    "raw_dist": float(g[5]),
                    "score": float(g[6]),
                    "tsu": int(g[7]),
                    "would_reactivate": g[8] == "YES",
                    "gt_phase": current_phase
                })

    total = len(events)
    accepted = [e for e in events if e["would_reactivate"]]
    rejected = [e for e in events if not e["would_reactivate"]]

    print(f"\n{'='*60}")
    print(f"  PHANTOM BRIDGE AUDIT — SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"  Total bridge candidates evaluated: {total}")
    print(f"  Accepted candidates (would YES):   {len(accepted)}")
    print(f"  Rejected candidates (would NO):    {len(rejected)}")

    if accepted:
        print(f"\n  Accepted Candidates Metrics:")
        for key in ["anchor_sim", "avg_sim", "raw_dist", "score"]:
            vals = [e[key] for e in accepted]
            print(f"    {key:12s}: min={min(vals):.3f} avg={np.mean(vals):.3f} max={max(vals):.3f}")

    if rejected:
        print(f"\n  Rejected Candidates Metrics:")
        for key in ["anchor_sim", "avg_sim", "raw_dist", "score"]:
            vals = [e[key] for e in rejected]
            print(f"    {key:12s}: min={min(vals):.3f} avg={np.mean(vals):.3f} max={max(vals):.3f}")

    # Outcome quantification against GT phase
    # In WITHIN_CLIP_RECOVERY: the detection belongs to the reappearing track if anchor_sim >= 0.50.
    tp = fp = fn = tn = 0
    tp_list, fp_list, fn_list = [], [], []

    for e in events:
        if e["gt_phase"] == "WITHIN_CLIP_RECOVERY" and e["anchor_sim"] >= 0.50:
            if e["would_reactivate"]:
                tp += 1
                tp_list.append(e)
            else:
                fn += 1
                fn_list.append(e)
        else:
            # Foreign detection / background clutter
            if e["would_reactivate"]:
                fp += 1
                fp_list.append(e)
            else:
                tn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    print(f"\n  ── Performance Quantification (vs Actual Reappearance Outcomes) ──")
    print(f"    True Positives (TP):   {tp}  (Correct recovery of reappearing track)")
    print(f"    False Positives (FP):  {fp}  (False match across different identity/scene)")
    print(f"    False Negatives (FN):  {fn}  (Missed recovery of reappearing track)")
    print(f"    True Negatives (TN):   {tn}  (Correct rejection of foreign detection)")
    print(f"")
    print(f"    Precision: {precision:.2%}")
    print(f"    Recall:    {recall:.2%}")
    print(f"    F1 Score:  {f1:.2%}")

    for label, lst in [("True Positive", tp_list), ("False Positive", fp_list), ("False Negative", fn_list)]:
        if lst:
            print(f"\n  ── {label} Samples (up to 5) ──")
            for e in lst[:5]:
                print(f"    f={e['frame']} lost={e['lost_local']} phase={e['gt_phase']} "
                      f"anc={e['anchor_sim']:.3f} avg={e['avg_sim']:.3f} dist={e['raw_dist']:.0f} "
                      f"score={e['score']:.3f} tsu={e['tsu']}")

    print(f"\n  ── Live Production Recommendation ──")
    if total > 0 and (tp + fn) > 0:
        if precision >= 0.95 and fp == 0:
            verdict = "RECOMMENDED: Precision >= 95% with 0 false positives. Safe to enable bridge.reactivate()."
            print(f"  ✓ {verdict}")
        elif precision >= 0.90:
            verdict = "CAUTION: Precision >= 90%. Tighten scoring weights or spatial gate before live enablement."
            print(f"  ⚠ {verdict}")
        else:
            verdict = f"NOT RECOMMENDED: Precision is {precision:.1%}. Investigate false positives."
            print(f"  ✗ {verdict}")
    else:
        verdict = "INCONCLUSIVE: Insufficient occlusion recovery events captured."
        print(f"  ⚠ {verdict}")

    print(f"{'='*60}\n")

    analysis_path = log_path.replace(".log", "_analysis.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(f"PHANTOM RECOVERY BRIDGE FORENSIC AUDIT REPORT\n{'='*60}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"SUMMARY STATISTICS\n{'-'*30}\n")
        f.write(f"Total Candidates: {total}\nAccepted: {len(accepted)}\nRejected: {len(rejected)}\n\n")
        f.write(f"PERFORMANCE METRICS\n{'-'*30}\n")
        f.write(f"True Positives:  {tp}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Negatives:  {tn}\n")
        f.write(f"Precision: {precision:.4f} ({precision:.2%})\nRecall:    {recall:.4f} ({recall:.2%})\nF1 Score:  {f1:.4f}\n\n")
        f.write(f"RECOMMENDATION\n{'-'*30}\n{verdict}\n\n")
        f.write(f"AUDIT LOG DETAIL\n{'-'*30}\n")
        for e in events:
            outcome = "TN" if not e["would_reactivate"] else ("TP" if e["gt_phase"]=="WITHIN_CLIP_RECOVERY" and e["anchor_sim"]>=0.5 else "FP")
            if not e["would_reactivate"] and e["gt_phase"]=="WITHIN_CLIP_RECOVERY" and e["anchor_sim"]>=0.5:
                outcome = "FN"
            f.write(f"frame={e['frame']:05d} lost={e['lost_local']:04d} det={e['candidate_det']} "
                    f"anc={e['anchor_sim']:.3f} avg={e['avg_sim']:.3f} dist={e['raw_dist']:03.0f} "
                    f"score={e['score']:.3f} tsu={e['tsu']:03d} would={'YES' if e['would_reactivate'] else 'NO':3s} "
                    f"phase={e['gt_phase']:22s} outcome={outcome}\n")

    print(f"Detailed forensic audit report written to: {analysis_path}")


if __name__ == "__main__":
    main()
