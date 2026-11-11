"""JSON-compatible observations. Coordinates are original-image pixels.

Class IDs are model-local: downstream code must use label + model provenance.
An absent capability is None, never a fabricated zero-valued observation.
"""
from dataclasses import asdict, dataclass
import math
import json


@dataclass(frozen=True)
class FrameRef:
    camera_id: str
    stream_epoch: str
    frame_index: int
    captured_at_s: float
    received_at_s: float
    width: int
    height: int
    source: str
    timestamp_method: str = "frame_index/fps+offset"

    def __post_init__(self):
        if not self.camera_id or not self.stream_epoch:
            raise ValueError("camera_id and stream_epoch are required")
        if self.frame_index < 0 or min(self.width, self.height) <= 0:
            raise ValueError("Invalid frame dimensions/index")
        if not all(math.isfinite(x) for x in (self.captured_at_s, self.received_at_s)):
            raise ValueError("Timestamps must be finite")

    @property
    def observation_id(self):
        return json.dumps([self.camera_id, self.stream_epoch, self.frame_index], separators=(",", ":"))


@dataclass(frozen=True)
class Instance:
    box_xyxy: tuple
    score: float
    class_id: int
    label: str
    mask_polygon_xy: tuple | None = None
    keypoints_xyc: tuple | None = None

    def __post_init__(self):
        if len(self.box_xyxy) != 4 or not all(math.isfinite(x) for x in self.box_xyxy):
            raise ValueError("Box must contain four finite pixel coordinates")
        x1, y1, x2, y2 = self.box_xyxy
        if x2 < x1 or y2 < y1 or not 0 <= self.score <= 1:
            raise ValueError("Invalid box or confidence")


@dataclass(frozen=True)
class Observation:
    frame: FrameRef
    instances: tuple[Instance, ...]
    model: str
    task: str
    inference_ms: float
    keypoint_schema: str | None = None
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)
