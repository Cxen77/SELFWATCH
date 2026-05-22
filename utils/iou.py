import cv2
import numpy as np

def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter + 1e-6)

def iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two lists of [x1,y1,x2,y2] boxes. Returns (M, N) array.
    
    Uses vectorized numpy broadcasting for O(1) Python overhead.
    """
    M, N = len(boxes_a), len(boxes_b)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    a = np.array(boxes_a, dtype=np.float32)  # (M, 4)
    b = np.array(boxes_b, dtype=np.float32)  # (N, 4)

    # Intersection corners via broadcasting: (M, 1, 4) vs (1, N, 4)
    ix1 = np.maximum(a[:, 0:1], b[:, 0].reshape(1, -1))  # (M, N)
    iy1 = np.maximum(a[:, 1:2], b[:, 1].reshape(1, -1))
    ix2 = np.minimum(a[:, 2:3], b[:, 2].reshape(1, -1))
    iy2 = np.minimum(a[:, 3:4], b[:, 3].reshape(1, -1))

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).reshape(-1, 1)  # (M, 1)
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).reshape(1, -1)  # (1, N)

    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)

def extract_histogram(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(64, dtype=np.float32)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)
