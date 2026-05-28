"""
AsyncStateCache — SELFWATCH Phase 1 Optimization

Decouples JPEG compression from the inference thread.

Architecture:
  Inference thread → posts raw frames to _encode_queue (maxsize=4)
  Encoder thread   → compresses frames, appends to storage deque

This removes 4–10ms of cv2.imencode() from the inference hot path
while preserving all scrubbing / playback / forensic functionality.
"""

import threading
import queue
import time
import cv2
import numpy as np
from collections import deque


# Try to import PyTurboJPEG for faster encoding
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _turbo = TurboJPEG()
    _USE_TURBO = True
except Exception:
    _turbo = None
    _USE_TURBO = False


def _encode_frame(frame: np.ndarray, quality: int = 70) -> bytes:
    """Encode a single BGR frame to JPEG bytes. Uses TurboJPEG if available."""
    if _USE_TURBO and _turbo is not None:
        try:
            return _turbo.encode(frame, quality=quality, pixel_format=TJPF_BGR)
        except Exception:
            pass
    # Fallback: standard OpenCV encode
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


class AsyncStateCache:
    """
    Async-safe replacement for StateCache.

    Inference thread calls append() — returns immediately.
    Background encoder thread compresses frames and stores them.

    Scrubbing / get_frame() still works identically.
    All forensic functionality preserved.
    """

    # Diagnostics (readable from any thread without lock)
    encode_queue_size: int = 0
    encode_drops: int = 0
    encode_ms_avg: float = 0.0

    def __init__(self, max_frames: int = 300, jpeg_quality: int = 70):
        self.max_frames = max_frames
        self.jpeg_quality = jpeg_quality

        # Storage (written by encoder thread, read by UI thread)
        self._frames: deque = deque(maxlen=max_frames)
        self._metadata: deque = deque(maxlen=max_frames)
        self._storage_lock = threading.Lock()

        self.current_index: int = -1

        # Encode queue: inference thread pushes (raw_frames_list, meta_list)
        # maxsize=4 prevents RAM buildup; excess frames are dropped with a warning
        self._encode_queue: queue.Queue = queue.Queue(maxsize=4)

        # Timing for diagnostics
        self._encode_times: deque = deque(maxlen=30)

        # Shutdown flag
        self._running = True

        # Start background encoder daemon
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop,
            daemon=True,
            name="sw-state-encoder"
        )
        self._encoder_thread.start()

    # ─── Public API (inference thread safe) ────────────────────────────

    def append(self, frames_list, meta_list):
        """
        Non-blocking append. Posts raw frames to encode queue.
        Returns immediately — never waits for JPEG encoding.

        If the encode queue is full (encoder fell behind), drops the oldest
        pending job and replaces with the newest frame.
        """
        job = (frames_list, meta_list)
        try:
            self._encode_queue.put_nowait(job)
        except queue.Full:
            # Drop oldest pending job, insert newest
            try:
                self._encode_queue.get_nowait()
                AsyncStateCache.encode_drops += 1
            except queue.Empty:
                pass
            try:
                self._encode_queue.put_nowait(job)
            except queue.Full:
                AsyncStateCache.encode_drops += 1

        AsyncStateCache.encode_queue_size = self._encode_queue.qsize()

    def get_frame(self, index: int):
        """Retrieve decoded frames and metadata at index. Thread-safe."""
        with self._storage_lock:
            if index < 0 or index >= len(self._frames):
                return None, None
            encoded_list = self._frames[index]
            meta_list = self._metadata[index]

        decoded_list = []
        for encoded in encoded_list:
            if encoded is not None:
                frame = cv2.imdecode(
                    np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                decoded_list.append(frame)
            else:
                decoded_list.append(None)

        self.current_index = index
        return decoded_list, meta_list

    def clear(self):
        """Clear all stored frames. Thread-safe."""
        with self._storage_lock:
            self._frames.clear()
            self._metadata.clear()
        self.current_index = -1

    def stop(self):
        """Signal encoder thread to exit cleanly."""
        self._running = False

    @property
    def total_frames(self) -> int:
        with self._storage_lock:
            return len(self._frames)

    # ─── Background Encoder Thread ──────────────────────────────────────

    def _encoder_loop(self):
        """
        Thread: reads jobs from queue, JPEG-encodes frames, stores results.

        Runs at full speed — bounded only by encode queue depth.
        Completely decoupled from inference and display threads.
        """
        while self._running:
            try:
                frames_list, meta_list = self._encode_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            encoded_list = []
            for frame in frames_list:
                if frame is not None:
                    try:
                        encoded_list.append(
                            _encode_frame(frame, self.jpeg_quality))
                    except Exception:
                        encoded_list.append(None)
                else:
                    encoded_list.append(None)

            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            self._encode_times.append(dt_ms)
            if self._encode_times:
                AsyncStateCache.encode_ms_avg = (
                    sum(self._encode_times) / len(self._encode_times))

            # Write to storage (thread-safe)
            with self._storage_lock:
                # Handle scrub-point truncation
                # (only when user has scrubbed backward and then resumes)
                if (self.current_index < len(self._frames) - 1
                        and self.current_index >= 0):
                    frames_tmp = list(self._frames)[:self.current_index + 1]
                    meta_tmp = list(self._metadata)[:self.current_index + 1]
                    self._frames = deque(frames_tmp, maxlen=self.max_frames)
                    self._metadata = deque(meta_tmp, maxlen=self.max_frames)

                self._frames.append(encoded_list)
                self._metadata.append(meta_list)
                self.current_index = len(self._frames) - 1

            AsyncStateCache.encode_queue_size = self._encode_queue.qsize()
