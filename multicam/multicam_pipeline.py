"""
Multi-Camera Pipeline Orchestrator — Phase 1

Orchestrates 2+ camera streams with synchronized frame processing,
shared global identity management, and cross-camera ReID.

Architecture:
    ┌──────────────┐   ┌──────────────┐
    │  Camera 0    │   │  Camera 1    │   ...
    │  (local trk) │   │  (local trk) │
    └──────┬───────┘   └──────┬───────┘
           │                  │
           ▼                  ▼
    ┌─────────────────────────────────┐
    │   Shared Global Identity Space  │
    │   (GlobalMultiCamIdentityMgr)   │
    └──────────────┬──────────────────┘
                   │
           ┌───────┴────────┐
           │  Cross-Camera   │
           │  ReID Matcher   │
           └────────────────┘

Processing modes:
    - SYNCHRONOUS: round-robin frame processing (default, simpler)
    - THREADED: each camera in its own thread (higher throughput)

Phase 1 uses SYNCHRONOUS mode for stability.
"""

import time
import cv2
import numpy as np
import threading
from typing import List, Dict, Optional, Any, Tuple

import config
from detectors import RTDETRDetector
from reid import EmbeddingExtractor
from trackers import StrongSORTTracker
from engine.pipeline import SelfWatchPipeline

from .global_registry import GlobalMultiCameraIdentityManager
from .camera_stream import CameraStream
from .cross_camera_reid import CrossCameraReIDMatcher
from .events import CameraEventBus, CameraEvent, EventType


class MultiCameraPipeline:
    """
    Orchestrates multiple camera streams with shared identity space.

    Usage:
        pipeline = MultiCameraPipeline()
        pipeline.add_camera(0)           # webcam 0
        pipeline.add_camera(1)           # webcam 1
        pipeline.add_camera("video.mp4") # video file
        pipeline.run()
    """

    def __init__(self,
                 detector_variant: str = "nano",
                 detector_resolution: int = 384,
                 use_fp16: bool = True,
                 similarity_threshold: float = 0.70,
                 max_dormant_time: float = 300.0,
                 enable_debug: bool = False):
        """
        Args:
            detector_variant: RT-DETR variant (nano/medium/large)
            detector_resolution: Detector input resolution
            use_fp16: Enable FP16 inference
            similarity_threshold: Cross-camera ReID threshold
            max_dormant_time: Max seconds for dormant identity survival
            enable_debug: Enable debug overlay
        """
        self._detector_variant = detector_variant
        self._detector_resolution = detector_resolution
        self._use_fp16 = use_fp16
        self._enable_debug = enable_debug

        # ── Shared components ───────────────────────────────────────────
        self.global_registry = GlobalMultiCameraIdentityManager(
            max_dormant=200,
            dormant_decay_rate=0.998,
        )
        self.event_bus = CameraEventBus(max_history=10000)
        self.cross_camera_matcher = CrossCameraReIDMatcher(
            global_registry=self.global_registry,
            event_bus=self.event_bus,
            similarity_threshold=similarity_threshold,
            max_time_gap=max_dormant_time,
        )

        # ── Per-camera streams ──────────────────────────────────────────
        self.cameras: List[CameraStream] = []
        self._camera_map: Dict[int, CameraStream] = {}

        # ── Shared detector and ReID extractor (GPU resources) ──────────
        # Single detector and ReID model shared across cameras
        # to avoid GPU memory duplication
        self._detector = None
        self._reid = None

        # ── Event logging listener ──────────────────────────────────────
        self.event_bus.register_listener(self._on_event)

        # ── Runtime state ───────────────────────────────────────────────
        self._running = False
        self._frame_count = 0
        self._start_time = 0.0
        self._last_results: Dict[int, Tuple] = {}  # cam_index -> last (frame, stats)

    # ═══════════════════════════════════════════════════════════════════
    #  SETUP
    # ═══════════════════════════════════════════════════════════════════

    def _ensure_shared_models(self):
        """
        Initialize shared GPU models (detector + ReID) once.
        All cameras share the same models to save GPU memory.
        """
        if self._detector is not None:
            return

        print(f"\n{'='*60}")
        print(f"  SELFWATCH Multi-Camera — Initializing Shared Models")
        print(f"{'='*60}")

        self._detector = RTDETRDetector(
            variant=self._detector_variant,
            resolution=self._detector_resolution,
            use_amp=self._use_fp16,
            compile_model=False,
        )

        self._reid = EmbeddingExtractor(
            weights_path=config.REID_WEIGHTS,
            device=self._detector.get_device(),
            half=config.REID_HALF,
        )

        self._detector.warmup()

        print(f"[MULTICAM] Shared detector: {self._detector.get_name()}")
        print(f"[MULTICAM] Shared ReID: OSNet x1.0 (512-dim)")
        print(f"{'='*60}\n")

    def add_camera(self, source, label: str = "",
                   camera_id: Optional[int] = None) -> int:
        """
        Add a camera to the multi-camera pipeline.

        Args:
            source: Video source (int for webcam, str for file/RTSP)
            label: Human-readable camera label
            camera_id: Explicit camera ID (auto-assigned if None)

        Returns:
            Assigned camera_id
        """
        self._ensure_shared_models()

        if camera_id is None:
            camera_id = len(self.cameras)

        # Each camera gets its own independent tracker and pipeline
        # but shares the detector and ReID extractor
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
            detector=self._detector,
            reid=self._reid,
            tracker=tracker,
            enable_debug_overlay=self._enable_debug,
        )

        stream = CameraStream(
            camera_id=camera_id,
            source=source,
            pipeline=pipeline,
            global_registry=self.global_registry,
            event_bus=self.event_bus,
            label=label or f"Camera{camera_id}",
        )

        self.cameras.append(stream)
        self._camera_map[camera_id] = stream

        print(f"[MULTICAM] Added Camera{camera_id}: {stream.label} "
              f"(source={source})")

        return camera_id

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN LOOP (SYNCHRONOUS)
    # ═══════════════════════════════════════════════════════════════════

    def run(self, max_fps: float = 15.0, display: bool = True,
            grid_layout: bool = True):
        """
        Main processing loop — synchronous round-robin.

        Processes one frame from each camera per iteration,
        then displays all frames in a grid or separate windows.

        Args:
            max_fps: Target FPS limit
            display: Show cv2 windows
            grid_layout: Combine all cameras into one grid window
        """
        frame_time_target = 1.0 / max_fps
        self._running = True
        self._start_time = time.time()

        # Open all cameras
        for cam in self.cameras:
            if not cam.open():
                print(f"[MULTICAM] WARNING: Camera{cam.camera_id} failed to open!")

        active_cameras = [c for c in self.cameras if c.is_open]
        if not active_cameras:
            print("[MULTICAM] ERROR: No cameras available!")
            return

        print(f"\n{'='*60}")
        print(f"  SELFWATCH Multi-Camera — Running")
        print(f"  Cameras: {len(active_cameras)}")
        print(f"  Target FPS: {max_fps}")
        print(f"  Press 'q' to quit")
        print(f"{'='*60}\n")

        try:
            while self._running:
                loop_start = time.perf_counter()
                self._frame_count += 1
                frames = []
                all_stats = []

                # ── Process each camera ─────────────────────────────────
                for cam in active_cameras:
                    ret, frame = cam.read_frame()
                    if not ret:
                        # End of stream for this camera
                        frames.append(None)
                        all_stats.append(None)
                        continue

                    processed_frame, stats = cam.process_frame(
                        frame,
                        cross_camera_matcher=self.cross_camera_matcher)

                    frames.append(processed_frame)
                    all_stats.append(stats)

                # ── Decay dormant identities ────────────────────────────
                if self._frame_count % 30 == 0:
                    self.global_registry.decay_dormant()

                # ── Check if all cameras exhausted ──────────────────────
                if all(f is None for f in frames):
                    print("[MULTICAM] All camera streams ended.")
                    break

                # ── Display ─────────────────────────────────────────────
                if display:
                    valid_frames = [(i, f) for i, f in enumerate(frames)
                                    if f is not None]

                    if grid_layout and len(valid_frames) > 1:
                        grid = self._build_grid(
                            [f for _, f in valid_frames],
                            [active_cameras[i].label for i, _ in valid_frames])
                        cv2.imshow("SELFWATCH Multi-Camera", grid)
                    else:
                        for i, f in valid_frames:
                            cam = active_cameras[i]
                            cv2.imshow(
                                f"SELFWATCH - {cam.label}", f)

                # ── Print status ────────────────────────────────────────
                if self._frame_count % 60 == 0:
                    self._print_status(active_cameras, all_stats)

                # ── Print per-frame profiling ───────────────────────────
                fps_strs = []
                for cam, stats in zip(active_cameras, all_stats):
                    if stats:
                        fps_strs.append(
                            f"CAM{cam.camera_id}:{cam.fps:.1f}fps")
                reg_stats = self.global_registry.get_stats()
                print(f"\r[MULTICAM] {' | '.join(fps_strs)} | "
                      f"GIDs:{reg_stats['active_global_ids']} "
                      f"Dormant:{reg_stats['dormant_count']} "
                      f"XCam:{reg_stats['total_cross_camera_matches']}",
                      end="")

                # ── Key handling ────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    for cam in active_cameras:
                        state = cam.pipeline.debug_overlay.toggle()
                        print(f"\n[DEBUG CAM{cam.camera_id}] "
                              f"Overlay {'ON' if state else 'OFF'}")

                # ── FPS limiter ─────────────────────────────────────────
                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_time_target:
                    time.sleep(frame_time_target - elapsed)

        except KeyboardInterrupt:
            print("\n[MULTICAM] Interrupted by user.")

        finally:
            self._shutdown(active_cameras)

    # ═══════════════════════════════════════════════════════════════════
    #  SINGLE STEP (for external orchestration)
    # ═══════════════════════════════════════════════════════════════════

    def step(self) -> Optional[List[Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]]]:
        """
        Process one frame from each camera (for external loop control).

        Sequential non-blocking batching:
          - Eliminates Python GIL contention and PyTorch CUDA context switching 
            overhead that caused 1.3 FPS drop with ThreadPoolExecutor.
          - Each camera gets up to 50ms to deliver a frame.
          - If a camera has no frame ready, reuse its last result.
          - A slow camera NEVER blocks a healthy camera.

        Returns list of (frame, stats) per camera, or None if ALL streams are ended.
        """
        self._frame_count += 1
        num_cams = len(self.cameras)
        results = []
        all_ended = True

        for i, cam in enumerate(self.cameras):
            if not cam.is_open and not cam._capture_running:
                results.append((None, None))
                continue

            if cam._stream_ended and len(cam._frame_buf) == 0:
                results.append((None, None))
                continue

            # If we reach here, this camera is still active
            all_ended = False

            # Non-blocking read with short timeout
            # Prevents head-of-line blocking from slow cameras
            ret, frame = cam.read_frame(timeout=0.05)

            if not ret and frame is None:
                # Stream ended permanently during read
                results.append((None, None))
                continue

            if frame is None:
                # Camera is alive but no new frame — reuse last result
                # This prevents stalling the entire pipeline
                last = self._last_results.get(i)
                if last is not None:
                    results.append(last)
                else:
                    results.append((None, None))
                continue

            # Process the frame sequentially (fastest for PyTorch + GIL)
            processed_frame, stats = cam.process_frame(
                frame, cross_camera_matcher=self.cross_camera_matcher)
            result = (processed_frame, stats)
            results.append(result)

            # Cache this result for frame-reuse
            self._last_results[i] = result

        if all_ended:
            return None

        # Periodic dormant decay
        if self._frame_count % 30 == 0:
            self.global_registry.decay_dormant()

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  DISPLAY HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_grid(frames: List[np.ndarray],
                    labels: List[str],
                    target_width: int = 640) -> np.ndarray:
        """
        Arrange frames into a 2-column grid for display.
        Resizes each frame to target_width maintaining aspect ratio.
        """
        n = len(frames)
        if n == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Resize all frames to same width
        resized = []
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            scale = target_width / w
            new_h = int(h * scale)
            r = cv2.resize(f, (target_width, new_h))
            resized.append(r)

        # Compute grid layout (2 columns)
        cols = min(2, n)
        rows = (n + cols - 1) // cols

        # Pad frames to same height per row
        max_h = max(f.shape[0] for f in resized)
        padded = []
        for f in resized:
            if f.shape[0] < max_h:
                pad = np.zeros((max_h - f.shape[0], target_width, 3),
                               dtype=np.uint8)
                f = np.vstack([f, pad])
            padded.append(f)

        # Fill with black if needed for incomplete rows
        while len(padded) % cols != 0:
            padded.append(
                np.zeros((max_h, target_width, 3), dtype=np.uint8))

        # Stack into grid
        row_images = []
        for r in range(rows):
            row = np.hstack(padded[r * cols: (r + 1) * cols])
            row_images.append(row)
        grid = np.vstack(row_images)

        return grid

    # ═══════════════════════════════════════════════════════════════════
    #  STATUS & LOGGING
    # ═══════════════════════════════════════════════════════════════════

    def _on_event(self, event: CameraEvent):
        """Handle events from the event bus (logging)."""
        if event.event_type in (EventType.ENTER, EventType.EXIT,
                                EventType.MATCH):
            print(f"\n  {event}")

    def _print_status(self, cameras, all_stats):
        """Print periodic status summary."""
        reg_stats = self.global_registry.get_stats()
        event_summary = self.event_bus.get_summary()
        match_stats = self.cross_camera_matcher.get_stats()

        print(f"\n\n{'='*60}")
        print(f"  SELFWATCH Multi-Camera Status — Frame {self._frame_count}")
        print(f"{'='*60}")

        for cam in cameras:
            status = cam.get_status()
            print(f"  Camera{status['camera_id']}: {status['label']} "
                  f"| FPS:{status['fps']} "
                  f"| Active:{status['active_tracks']} "
                  f"| Total IDs:{status['total_identities']}")

        print(f"\n  Global Registry:")
        print(f"    Active GIDs: {reg_stats['active_global_ids']}")
        print(f"    Dormant: {reg_stats['dormant_count']}")
        print(f"    Total created: {reg_stats['total_global_ids_created']}")
        print(f"    Cross-camera matches: {reg_stats['total_cross_camera_matches']}")
        print(f"    Dormant recoveries: {reg_stats['total_dormant_recoveries']}")

        print(f"\n  Events: {event_summary}")
        print(f"  ReID Matcher: {match_stats}")
        print(f"{'='*60}\n")

    def _shutdown(self, cameras):
        """Graceful shutdown of all cameras."""
        print(f"\n\n{'='*60}")
        print(f"  SELFWATCH Multi-Camera — Shutting Down")
        print(f"{'='*60}")

        runtime = time.time() - self._start_time
        print(f"  Runtime: {runtime:.1f}s")
        print(f"  Total frames: {self._frame_count}")

        # Print final global registry status
        print(f"\n{self.global_registry.get_debug_summary()}")

        # Print event summary
        print(f"\n  Event Summary: {self.event_bus.get_summary()}")

        # Print transition pairs
        pairs = self.event_bus.get_transition_pairs()
        if pairs:
            print(f"\n  Cross-Camera Transitions ({len(pairs)}):")
            for exit_e, enter_e in pairs[-10:]:
                gap = enter_e.timestamp - exit_e.timestamp
                print(f"    GID {exit_e.global_id}: "
                      f"Camera{exit_e.camera_id} -> Camera{enter_e.camera_id} "
                      f"(gap={gap:.1f}s)")

        # Print ReID matcher stats
        print(f"\n  ReID Matcher: {self.cross_camera_matcher.get_stats()}")

        # Close all cameras
        for cam in cameras:
            cam.close()

        cv2.destroyAllWindows()
        self._running = False
        print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def get_camera(self, camera_id: int) -> Optional[CameraStream]:
        """Get a camera stream by ID."""
        return self._camera_map.get(camera_id)

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def is_running(self) -> bool:
        return self._running
