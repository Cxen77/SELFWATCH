"""Bounded, event-time ordered replay. CFR timestamps are explicit estimates.

This runner does not claim hardware synchronization or correct VFR timestamps.
Offsets must come from the recording protocol/calibration. One frame per source
is buffered; frames from different cameras are not assumed simultaneous.
"""
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import time

from .contracts import FrameRef


@dataclass(frozen=True)
class Source:
    camera_id: str
    path: Path
    stream_epoch: str
    fps: float
    offset_s: float = 0.0
    timestamps_s: tuple[float, ...] | None = None

    def __post_init__(self):
        if not self.camera_id or not self.stream_epoch:
            raise ValueError("Camera ID and recording epoch are required")
        if not math.isfinite(self.fps) or self.fps <= 0 or not math.isfinite(self.offset_s):
            raise ValueError("Finite positive FPS and finite offset required")
        if not self.path.is_file():
            raise ValueError(f"Recording does not exist: {self.path}")
        if self.timestamps_s is not None:
            if not self.timestamps_s or not all(math.isfinite(t) for t in self.timestamps_s):
                raise ValueError("Explicit timestamps must be nonempty and finite")
            if any(b <= a for a, b in zip(self.timestamps_s, self.timestamps_s[1:])):
                raise ValueError("Explicit timestamps must increase strictly")

    def timestamp(self, index):
        return self.offset_s + (self.timestamps_s[index] if self.timestamps_s is not None else index / self.fps)


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not data.get("sources"):
        raise ValueError("Expected schema_version 1 and nonempty sources")
    sources = [Source(str(s["camera_id"]), (path.parent / s["path"]).resolve(),
                      str(s["stream_epoch"]), float(s["fps"]), float(s.get("offset_s", 0)),
                      tuple(float(t) for t in s["timestamps_s"]) if "timestamps_s" in s else None)
               for s in data["sources"]]
    if len({s.camera_id for s in sources}) != len(sources):
        raise ValueError("A replay manifest must have unique camera IDs")
    return sources


def ordered_merge(streams):
    """Merge (FrameRef, image) streams with at most one pending frame per input."""
    heap = []
    for number, stream in enumerate(streams):
        iterator = iter(stream)
        item = next(iterator, None)
        if item is not None:
            heapq.heappush(heap, (item[0].captured_at_s, number, item, iterator))
    while heap:
        timestamp, number, item, iterator = heapq.heappop(heap)
        yield item
        following = next(iterator, None)
        if following is not None:
            if following[0].captured_at_s < timestamp:
                raise ValueError("Source capture timestamps went backwards")
            heapq.heappush(heap, (following[0].captured_at_s, number, following, iterator))


def decode(source):
    import cv2
    capture = cv2.VideoCapture(str(source.path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open recording: {source.path}")
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if source.timestamps_s is not None and index >= len(source.timestamps_s):
                raise ValueError("Video has more frames than explicit timestamps")
            height, width = image.shape[:2]
            yield FrameRef(source.camera_id, source.stream_epoch, index,
                source.timestamp(index), time.time(), width, height,
                str(source.path), "dataset_frame_timestamps+offset" if source.timestamps_s is not None
                else "frame_index/fps+offset"), image
            index += 1
        if index == 0:
            raise RuntimeError(f"No decodable frames: {source.path}")
        if source.timestamps_s is not None and index != len(source.timestamps_s):
            raise ValueError("Video has fewer frames than explicit timestamps")
    finally:
        capture.release()


def replay(sources):
    streams = [decode(source) for source in sources]
    try:
        yield from ordered_merge(streams)
    finally:
        for stream in streams:
            stream.close()
