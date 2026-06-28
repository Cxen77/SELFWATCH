"""
Phase B Detector Audit Script
Measures detection counts, confidences, and FPS impact
when expanding target_classes from person-only to mixed.
"""
import sys, time, os, cv2
# Ensure project root is in sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import numpy as np
from collections import defaultdict

COCO_NAMES = {
    0: "person", 26: "backpack", 30: "handbag",
    32: "suitcase", 72: "laptop", 76: "cell phone",
}
PERSON_ONLY = [0]
MIXED_CLASSES = [0, 26, 30, 32, 72, 76]
SEP = "=" * 60
DASH = "-" * 45

def run_audit(detector, cap, target_classes, label, max_frames=200):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    counts = defaultdict(int)
    conf_sums = defaultdict(float)
    total_dets = 0
    frame_times = []

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (960, 540))
        t0 = time.perf_counter()
        result = detector.detect(frame, conf_threshold=0.35, target_classes=target_classes)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        for cid, sc in zip(result.class_ids, result.scores):
            cid = int(cid)
            counts[cid] += 1
            conf_sums[cid] += float(sc)
            total_dets += 1

    avg_ms = np.mean(frame_times) * 1000
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    n_frames = len(frame_times)

    print("")
    print(SEP)
    print("  DETECTOR AUDIT: " + label)
    print("  target_classes=" + str(target_classes))
    print("  Frames: %d  |  Avg detect: %.1fms  |  Det FPS: %.1f" % (n_frames, avg_ms, avg_fps))
    print(SEP)
    print("  %-15s %8s %10s %10s" % ("Class", "Count", "Avg/Frame", "Avg Conf"))
    print("  " + DASH)
    for cid in sorted(counts.keys()):
        name = COCO_NAMES.get(cid, "cls%d" % cid)
        avg_per_frame = counts[cid] / n_frames
        avg_conf = conf_sums[cid] / counts[cid] if counts[cid] > 0 else 0
        print("  %-15s %8d %10.1f %10.3f" % (name, counts[cid], avg_per_frame, avg_conf))
    print("  " + DASH)
    print("  %-15s %8d %10.1f" % ("TOTAL", total_dets, total_dets / n_frames))
    print(SEP)
    return avg_ms, counts


if __name__ == "__main__":
    import config
    from detectors.rtdetr_detector import RTDETRDetector

    detector = RTDETRDetector(
        variant=config.DETECTOR_VARIANT,
        resolution=config.DETECTOR_RESOLUTION,
        use_amp=config.DETECTOR_AMP,
        compile_model=config.DETECTOR_COMPILE,
    )
    detector.warmup()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera 0")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    audit_frames = min(200, total_frames if total_frames > 0 else 200)
    print("Video source: camera 0, auditing %d frames" % audit_frames)

    ms1, _ = run_audit(detector, cap, PERSON_ONLY, "PERSON-ONLY", audit_frames)
    ms2, _ = run_audit(detector, cap, MIXED_CLASSES, "MIXED (Person+Objects)", audit_frames)

    delta = ms2 - ms1
    print("")
    print(SEP)
    print("  FPS IMPACT SUMMARY")
    print("  Person-only detect: %.1fms" % ms1)
    print("  Mixed detect:       %.1fms" % ms2)
    print("  Delta:              %+.1fms (%+.1f%%)" % (delta, delta / ms1 * 100))
    print(SEP)

    cap.release()
