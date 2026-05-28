"""
SELFWATCH Multi-Camera Extension — Phase 1

Foundational infrastructure for running 2+ video streams
simultaneously with ONE shared global identity space.

Architecture:
    Camera Pipelines (independent local trackers)
        ↓
    Shared Global Memory (GlobalMultiCameraIdentityManager)
        ↓
    Cross-Camera Identity Assignment (baseline cosine ReID)
        ↓
    Future Cognitive Layer (Phase 2+)
"""

from .global_registry import GlobalMultiCameraIdentityManager
from .camera_stream import CameraStream
from .cross_camera_reid import CrossCameraReIDMatcher
from .events import CameraEventBus, CameraEvent
from .multicam_pipeline import MultiCameraPipeline

__all__ = [
    "GlobalMultiCameraIdentityManager",
    "CameraStream",
    "CrossCameraReIDMatcher",
    "CameraEventBus",
    "CameraEvent",
    "MultiCameraPipeline",
]
