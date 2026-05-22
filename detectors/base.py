"""
Base detector interface for the SELFWATCH pipeline.

All detector backends (YOLO, RF-DETR, etc.) must implement this interface
so they can be plugged into the tracking pipeline interchangeably.

Output format per detection: [x1, y1, x2, y2, confidence, class_id]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    """Unified detection output compatible with ByteTrack / PersistentTracker."""
    boxes: np.ndarray        # (N, 4)  — [x1, y1, x2, y2]
    scores: np.ndarray       # (N,)    — confidence scores
    class_ids: np.ndarray    # (N,)    — integer class IDs
    labels: List[str]        # (N,)    — human-readable class names

    @property
    def count(self) -> int:
        return len(self.scores)

    def filter_by_class(self, class_id: int) -> "DetectionResult":
        """Return only detections matching a specific class ID."""
        mask = self.class_ids == class_id
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            labels=[l for l, m in zip(self.labels, mask) if m],
        )

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Return only detections above a confidence threshold."""
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            labels=[l for l, m in zip(self.labels, mask) if m],
        )

    @staticmethod
    def empty() -> "DetectionResult":
        return DetectionResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            class_ids=np.empty(0, dtype=np.int32),
            labels=[],
        )


class BaseDetector(ABC):
    """
    Abstract detector interface.

    Subclasses must implement:
        - detect(frame) -> DetectionResult
        - warmup()      -> None  (optional, for JIT / compile warmup)
    """

    @abstractmethod
    def detect(self, frame: np.ndarray,
               conf_threshold: float = 0.3,
               target_classes: Optional[List[int]] = None) -> DetectionResult:
        """
        Run detection on a single BGR frame (numpy HWC uint8).

        Args:
            frame: BGR image as numpy array (H, W, 3), uint8.
            conf_threshold: Minimum confidence to keep a detection.
            target_classes: If set, only return detections of these class IDs.

        Returns:
            DetectionResult with boxes in [x1, y1, x2, y2] format.
        """
        ...

    def warmup(self, input_size: tuple = (640, 640)) -> None:
        """Optional warmup pass (useful for JIT / torch.compile)."""
        pass

    @abstractmethod
    def get_device(self) -> str:
        """Return the device string ('cuda:0', 'cpu', etc.)."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for logging."""
        ...
