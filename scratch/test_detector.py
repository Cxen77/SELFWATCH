import cv2
import numpy as np
from detectors.rtdetr_detector import RTDETRDetector, _RFDETR_COCO_NAMES

def main():
    detector = RTDETRDetector(variant="large", resolution=640)
    detector.warmup()

    cap = cv2.VideoCapture("detection_output.mp4")
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
        
    print("Frame read. Running raw detection...")
    
    # We want ALL classes, threshold 0.05
    res = detector.detect(frame, conf_threshold=0.05, target_classes=None)
    
    print(f"Total detections > 0.05: {len(res.boxes)}")
    
    # Print the distribution of classes and their confidences
    from collections import defaultdict
    class_confs = defaultdict(list)
    
    for box, score, class_id, label in zip(res.boxes, res.scores, res.class_ids, res.labels):
        class_confs[label].append(score)
        
    print("\n--- Class Confidence Distribution ---")
    for label, scores in sorted(class_confs.items(), key=lambda x: max(x[1]), reverse=True):
        scores_sorted = sorted(scores, reverse=True)
        top_3 = [f"{s:.3f}" for s in scores_sorted[:3]]
        print(f"{label:15s}: max={scores_sorted[0]:.3f} count={len(scores)} top_scores={top_3}")
        
    print("\n--- Detailed Detections ---")
    for box, score, class_id, label in zip(res.boxes, res.scores, res.class_ids, res.labels):
        if score > 0.15:
            print(f"Conf: {score:.3f} | Class: {label} (id: {class_id}) | Box: {box}")

if __name__ == "__main__":
    main()
