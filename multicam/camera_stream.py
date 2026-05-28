"""
Camera Stream — Per-Camera Processing Unit

Each CameraStream wraps:
    - One video source (webcam, RTSP, file)
    - One local SelfWatchPipeline (detector + tracker + memory)
    - One set of local track IDs
    - Local frame timeline

But shares:
    - GlobalMultiCameraIdentityManager (cross-camera identity)
    - CameraEventBus (entry/exit events)
    - CrossCameraReIDMatcher (dormant matching)

Architecture (stabilized):
    Thread A (capture): reads frames, keeps only latest in deque(maxlen=1)
    Thread B (inference): called externally, pops latest frame non-blocking
    No blocking waits. No unbounded queues. Constant RAM.
"""

import time
import threading
import cv2
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Set, Tuple

from .events import CameraEventBus, CameraEvent, EventType
from .global_registry import GlobalMultiCameraIdentityManager


class CameraStream:
    """
    Per-camera processing unit.

    Manages one video source and its local tracking pipeline,
    while interfacing with the shared global identity system.
    """

    def __init__(self,
                 camera_id: int,
                 source,
                 pipeline,
                 global_registry: GlobalMultiCameraIdentityManager,
                 event_bus: CameraEventBus,
                 label: str = ""):
        """
        Args:
            camera_id: Unique integer ID for this camera
            source: cv2.VideoCapture source (int for webcam, str for file/RTSP)
            pipeline: SelfWatchPipeline instance (independent per camera)
            global_registry: Shared global identity manager
            event_bus: Shared event bus
            label: Human-readable label (e.g., "Front Door", "Lobby")
        """
        self.camera_id = camera_id
        self.source = source
        self.pipeline = pipeline
        self.global_registry = global_registry
        self.event_bus = event_bus
        self.label = label or f"Camera{camera_id}"

        # Video capture
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

        # ── Async capture state ─────────────────────────────────────
        # Latest-frame queue: deque(maxlen=2) for 1-frame look-ahead
        self._frame_buf: deque = deque(maxlen=2)
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running = False
        self._stream_ended = False

        # Frame tracking
        self.frame_index = 0
        self.frame_time = 0.0
        self._last_frame_time = 0.0

        # Local state tracking
        # Tracks the mapping: local_pipeline_gid → multicam_global_id
        self._pipeline_gid_to_global: Dict[int, int] = {}
        # Set of local pipeline GIDs that were active last frame
        self._prev_active_gids: Set[int] = set()

        # Statistics
        self.total_frames = 0
        self.total_identities_seen = 0
        self.fps = 0.0
        self.dropped_frames = 0

    # ═══════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════

    def open(self) -> bool:
        """Open the video source and start the async capture thread."""
        self._cap = cv2.VideoCapture(self.source)
        self._is_open = self._cap.isOpened()

        if self._is_open:
            print(f"[CAM{self.camera_id}] Opened: {self.label} (source={self.source})")
            self._start_capture()
        else:
            print(f"[CAM{self.camera_id}] FAILED to open: {self.label} (source={self.source})")
        return self._is_open

    def _start_capture(self):
        """Start the background capture thread."""
        self._capture_running = True
        self._stream_ended = False
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True,
            name=f"cam{self.camera_id}-capture"
        )
        self._capture_thread.start()

    def _capture_loop(self):
        """
        Thread A: Continuously reads frames into deque(maxlen=2).

        Two modes:
          LIVE STREAMS: read as fast as possible, deque auto-evicts old.
            Inference always gets freshest frame. Old frames dropped.

          FILE SOURCES: back-pressure pacing. After placing a frame,
            wait until inference consumes it before reading the next.
            This ensures:
              - the video file is NOT exhausted prematurely
              - every frame is available for processing
              - no CPU burn from unnecessary decoding
              - video plays at inference speed, not decode speed
        """
        is_file = isinstance(self.source, str) and not (
            self.source.startswith('rtsp://') or
            self.source.startswith('http://') or
            self.source.startswith('https://')
        )

        while self._capture_running:
            if is_file:
                # BACK-PRESSURE: wait until inference has consumed the frame
                # before reading the next one from the file
                while len(self._frame_buf) >= 1 and self._capture_running:
                    time.sleep(0.005)
                if not self._capture_running:
                    break

            ret, frame = self._cap.read()
            if not ret:
                self._stream_ended = True
                self._capture_running = False
                break

            # Track how many frames we overwrote (dropped) — live streams only
            if len(self._frame_buf) > 0:
                self.dropped_frames += 1

            # deque auto-evicts old frame when full
            self._frame_buf.append(frame)
            self.total_frames += 1

    def close(self):
        """Release video source and pipeline resources."""
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        if self._cap is not None:
            self._cap.release()
        self.pipeline.close()
        self._is_open = False
        print(f"[CAM{self.camera_id}] Closed: {self.label}")

    @property
    def is_open(self) -> bool:
        return self._is_open and self._cap is not None and self._cap.isOpened()

    # ═══════════════════════════════════════════════════════════════════
    #  FRAME ACQUISITION (non-blocking)
    # ═══════════════════════════════════════════════════════════════════

    def read_frame(self, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Non-blocking read of the latest captured frame.

        Returns (True, frame) if a frame is available.
        Returns (False, None) if stream ended.
        Returns (True, None) if no frame ready yet (skip this camera).

        The timeout prevents indefinite blocking — if no frame arrives
        within timeout, we return (True, None) so the pipeline can
        continue processing other cameras.
        """
        if self._stream_ended and len(self._frame_buf) == 0:
            return False, None

        if not self.is_open and not self._capture_running:
            return False, None

        # Try to get the latest frame, with brief wait
        deadline = time.monotonic() + timeout
        frame = None
        while time.monotonic() < deadline:
            if len(self._frame_buf) > 0:
                try:
                    frame = self._frame_buf.pop()
                    # Clear any remaining (shouldn't happen with maxlen=1, but safe)
                    self._frame_buf.clear()
                except IndexError:
                    pass
                break
            if self._stream_ended:
                return False, None
            time.sleep(0.002)

        if frame is not None:
            self.frame_index += 1
            self._last_frame_time = self.frame_time
            self.frame_time = time.time()
            return True, frame

        # Timeout — no frame available yet, don't block
        return True, None

    # ═══════════════════════════════════════════════════════════════════
    #  FRAME PROCESSING (tracking + identity sync)
    # ═══════════════════════════════════════════════════════════════════

    def process_frame(self, frame: np.ndarray,
                      cross_camera_matcher=None,
                      _precomputed_det=None,
                      _precomputed_emb=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run one frame through the local pipeline and synchronize
        with the global identity system.

        Args:
            frame: BGR image
            cross_camera_matcher: CrossCameraReIDMatcher (optional)
            _precomputed_det: DetectionResult from batched GPU pass (Phase 5).
                              When provided, skips internal detector call.
            _precomputed_emb: np.ndarray (N,512) from batched ReID (Phase 5).
                              When provided, skips internal ReID call.

        Returns:
            (annotated_frame, stats_dict)
        """
        t0 = time.perf_counter()

        # ── Run local pipeline ──────────────────────────────────────────
        if _precomputed_det is not None or _precomputed_emb is not None:
            # Phase 5 batched path: inject pre-computed GPU results
            processed_frame, stats = self.pipeline.process_frame_with_precomputed(
                frame,
                det_result=_precomputed_det,
                embeddings=_precomputed_emb,
                frame_delta=1, frame_index=self.frame_index,
                color_map=self._pipeline_gid_to_global)
        else:
            # Normal single-camera path (unchanged)
            processed_frame, stats = self.pipeline.process_frame(
                frame, frame_delta=1, frame_index=self.frame_index,
                color_map=self._pipeline_gid_to_global)

        # ── Synchronize with global identity system ─────────────────────
        current_active_gids = set()
        active_dict = stats.get("active_dict", {})
        track_states = stats.get("track_states", {})

        for pipeline_gid, box in active_dict.items():
            current_active_gids.add(pipeline_gid)

            # Get embedding for this track (from pipeline's tracker)
            embedding = None
            velocity = None
            for track in self.pipeline.tracker.tracks:
                local_gid = self.pipeline.global_id_manager.get_global_id(
                    track.local_id)
                if local_gid == pipeline_gid:
                    embedding = track.get_averaged_embedding()
                    velocity = track.vel.tolist() if hasattr(track, 'vel') else None
                    break

            # Check if this pipeline GID already has a global mapping
            if pipeline_gid not in self._pipeline_gid_to_global:
                # New identity in this camera!
                # Attempt cross-camera matching first
                matched_global_id = None
                if cross_camera_matcher is not None and embedding is not None:
                    matched_global_id = cross_camera_matcher.attempt_match(
                        self.camera_id, pipeline_gid, embedding,
                        self.frame_index)

                if matched_global_id is not None:
                    # Cross-camera match found — reuse global ID
                    global_id = self.global_registry.register_local_track(
                        self.camera_id, pipeline_gid,
                        global_id=matched_global_id,
                        embedding=embedding,
                        box=list(box) if box is not None else None,
                        velocity=np.array(velocity) if velocity else None)
                    self.global_registry.reactivate_dormant(matched_global_id)
                else:
                    # No match — allocate new global ID
                    global_id = self.global_registry.register_local_track(
                        self.camera_id, pipeline_gid,
                        embedding=embedding,
                        box=list(box) if box is not None else None,
                        velocity=np.array(velocity) if velocity else None)

                    # Publish NEW_GLOBAL event
                    self.event_bus.publish(CameraEvent(
                        event_type=EventType.NEW_GLOBAL,
                        global_id=global_id,
                        camera_id=self.camera_id,
                        timestamp=time.time(),
                        frame_index=self.frame_index,
                        local_track_id=pipeline_gid,
                    ))

                self._pipeline_gid_to_global[pipeline_gid] = global_id
                self.total_identities_seen += 1

                # Publish ENTER event
                self.event_bus.publish(CameraEvent(
                    event_type=EventType.ENTER,
                    global_id=self._pipeline_gid_to_global[pipeline_gid],
                    camera_id=self.camera_id,
                    timestamp=time.time(),
                    frame_index=self.frame_index,
                    local_track_id=pipeline_gid,
                    embedding=embedding,
                ))
            else:
                # Existing identity — update observation
                global_id = self._pipeline_gid_to_global[pipeline_gid]
                self.global_registry.update_observation(
                    self.camera_id, global_id,
                    embedding=embedding,
                    box=list(box) if box is not None else None,
                    velocity=np.array(velocity) if velocity else None,
                    frame_index=self.frame_index)

        # ── Detect exits (was active last frame, not active now) ────────
        exited_gids = self._prev_active_gids - current_active_gids
        for pipeline_gid in exited_gids:
            global_id = self._pipeline_gid_to_global.get(pipeline_gid)
            if global_id is not None:
                # Publish EXIT event
                self.event_bus.publish(CameraEvent(
                    event_type=EventType.EXIT,
                    global_id=global_id,
                    camera_id=self.camera_id,
                    timestamp=time.time(),
                    frame_index=self.frame_index,
                    local_track_id=pipeline_gid,
                ))

                # Unregister from global registry
                self.global_registry.unregister_local_track(
                    self.camera_id, pipeline_gid)

                # Move to dormant if not active elsewhere
                self.global_registry.move_to_dormant(
                    global_id, self.camera_id, self.frame_index)

                # Clean up local mapping
                del self._pipeline_gid_to_global[pipeline_gid]

        self._prev_active_gids = current_active_gids

        # ── Compute FPS ─────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        self.fps = 1.0 / max(elapsed, 1e-6)

        # ── Annotate with multi-camera info ─────────────────────────────
        processed_frame = self._annotate_multicam(
            processed_frame, active_dict, stats)

        # ── Add multicam stats ──────────────────────────────────────────
        stats["camera_id"] = self.camera_id
        stats["camera_label"] = self.label
        stats["camera_fps"] = self.fps
        stats["dropped_frames"] = self.dropped_frames
        stats["multicam_active_gids"] = {
            pgid: self._pipeline_gid_to_global.get(pgid)
            for pgid in current_active_gids
        }

        return processed_frame, stats

    def _annotate_multicam(self, frame: np.ndarray,
                           active_dict: dict,
                           stats: dict) -> np.ndarray:
        """
        Phase 3: Overlay logic decoupled to UI thread.
        This function now just passes the frame through untouched.
        """
        return frame

    # ═══════════════════════════════════════════════════════════════════
    #  STATUS
    # ═══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Get camera status summary."""
        return {
            "camera_id": self.camera_id,
            "label": self.label,
            "is_open": self.is_open,
            "frame_index": self.frame_index,
            "total_frames": self.total_frames,
            "total_identities": self.total_identities_seen,
            "active_tracks": len(self._prev_active_gids),
            "fps": round(self.fps, 1),
            "dropped_frames": self.dropped_frames,
            "queue_size": len(self._frame_buf),
        }
