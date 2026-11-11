"""Ultralytics adapter: detection, instance polygons or pose per checkpoint.

No forced COCO ID remapping and no implied hand/pose capability for detectors.
"""
from hashlib import sha256
from pathlib import Path
import time

from .contracts import Instance, Observation


def convert_result(result, frame, model, elapsed_ms, task, keypoint_schema=None,include_masks=True):
    instances = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        polygons = result.masks.xy if include_masks and result.masks is not None else None
        keypoints = result.keypoints.data.cpu().tolist() if result.keypoints is not None else None
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            points = keypoints[i] if keypoints is not None else None
            # Coordinates without confidence cannot be promoted to confidence=1.
            if points is not None and any(len(point) != 3 for point in points):
                raise ValueError("Pose checkpoint must supply x/y/confidence triples")
            expected = {"coco17": 17, "coco-wholebody133": 133}.get(keypoint_schema)
            if points is not None and expected is not None and len(points) != expected:
                raise ValueError(f"Checkpoint has {len(points)} landmarks, incompatible with {keypoint_schema}")
            instances.append(Instance(tuple(box), score, int(cls), str(result.names[int(cls)]),
                tuple(tuple(map(float, p)) for p in polygons[i]) if polygons is not None else None,
                tuple(tuple(p) for p in points) if points is not None else None))
    return Observation(frame, tuple(instances), model, task, elapsed_ms, keypoint_schema)


class YoloPerception:
    def __init__(self, weights, device="cpu", imgsz=640, confidence=0.25, keypoint_schema=None, half=False):
        from ultralytics import YOLO
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise ValueError(f"Explicit local checkpoint required: {weights}")
        if not 0 < confidence <= 1 or imgsz < 32:
            raise ValueError("Invalid confidence or image size")
        self.model = YOLO(str(weights))
        if self.model.task not in ("detect", "segment", "pose"):
            raise ValueError(f"Unsupported checkpoint task: {self.model.task}")
        if self.model.task == "pose" and not keypoint_schema:
            raise ValueError("Pose requires --keypoint-schema matching the checkpoint")
        self.device, self.imgsz, self.confidence = device, imgsz, confidence
        self.half = half
        self.include_masks=True
        self.keypoint_schema = keypoint_schema
        self.provenance = {"checkpoint": str(weights), "sha256": sha256(weights.read_bytes()).hexdigest(),
                           "task": self.model.task, "names": self.model.names, "half": half}

    def predict(self, frame, image):
        start = time.perf_counter()
        precision = {'quantize': 16} if self.half else {}
        result = self.model.predict(image, device=self.device, imgsz=self.imgsz,
                                    conf=self.confidence, verbose=False, **precision)[0]
        # Materialize GPU tensors before stopping the timer (includes transfers).
        observation = convert_result(result, frame, self.provenance["sha256"], 0,
                                     'detect' if self.model.task=='segment' and not self.include_masks else self.model.task,
                                     self.keypoint_schema,self.include_masks)
        from dataclasses import replace
        return replace(observation, inference_ms=(time.perf_counter() - start) * 1000), result
