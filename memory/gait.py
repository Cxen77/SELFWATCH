"""
SELFWATCH - Gait Signature from Bounding Box Dynamics

Extracts a second, completely appearance-independent identity signal
from HOW a person moves. Uses FFT on bounding box height oscillations
to capture stride frequency, amplitude, and phase — unique per person.

The key insight: when a person walks, their head oscillates vertically
with a periodic signal (~1.5-2.5 Hz). This oscillation is visible in
the bounding box height time series. The frequency and amplitude of
this oscillation is biomechanically unique to each person's stride.

This provides a modality that works when visual appearance fails:
  - Same clothing
  - Lighting changes
  - Camera angle shifts
  - Partial occlusions

Cost: One 45-point FFT per track per extraction (~0.01ms).
"""

import numpy as np
from collections import deque


# Minimum frames needed before gait extraction is meaningful
MIN_GAIT_FRAMES = 30
# Sliding window size (~1.5 sec at 30fps)
GAIT_WINDOW = 45
# Gait feature dimensionality
GAIT_DIM = 8


class GaitSignature:
    """
    Extracts and stores gait signatures from bounding box dynamics.

    For each tracked person, maintains a sliding window of bbox
    measurements. When enough data is collected, extracts an 8-dim
    gait feature vector using FFT analysis.

    The gait vector captures:
      [0] dominant_frequency    — stride rate (Hz, assuming 15fps)
      [1] frequency_power       — strength of the periodic signal
      [2] secondary_frequency   — second harmonic
      [3] height_variance       — variability in bbox height
      [4] width_variance        — variability in bbox width
      [5] speed_mean            — average movement speed
      [6] speed_variance        — movement speed consistency
      [7] aspect_ratio_mean     — average body proportions

    Usage:
        gait = GaitSignature(fps_estimate=15)

        # Every frame, feed bbox data:
        gait.update(track_id, bbox)

        # When retrieving identity:
        sig_a = gait.get_signature(track_id_a)
        sig_b = gait.get_signature(track_id_b)
        sim = gait.compare(sig_a, sig_b)  # 0.0 to 1.0
    """

    def __init__(self, fps_estimate=15):
        self.fps_estimate = fps_estimate
        self._windows = {}        # track_id -> deque of (cx, cy, w, h)
        self._signatures = {}     # track_id -> 8-dim numpy array

    def update(self, track_id, bbox):
        """
        Feed a new bounding box observation for a track.

        Args:
            track_id: Track ID.
            bbox:     [x1, y1, x2, y2] bounding box.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if track_id not in self._windows:
            self._windows[track_id] = deque(maxlen=GAIT_WINDOW)

        self._windows[track_id].append((cx, cy, w, h))

        # Extract signature when we have enough data
        if len(self._windows[track_id]) >= MIN_GAIT_FRAMES:
            self._extract(track_id)

    def _extract(self, track_id):
        """Extract gait signature from the sliding window."""
        data = list(self._windows[track_id])
        n = len(data)

        cx_arr = np.array([d[0] for d in data], dtype=np.float32)
        cy_arr = np.array([d[1] for d in data], dtype=np.float32)
        w_arr = np.array([d[2] for d in data], dtype=np.float32)
        h_arr = np.array([d[3] for d in data], dtype=np.float32)

        # FFT on height signal (captures head oscillation)
        h_detrended = h_arr - np.mean(h_arr)
        fft_result = np.fft.rfft(h_detrended)
        power = np.abs(fft_result)[1:]   # Skip DC component
        freqs = np.fft.rfftfreq(n, d=1.0/self.fps_estimate)[1:]

        if len(power) == 0:
            self._signatures[track_id] = np.zeros(GAIT_DIM, dtype=np.float32)
            return

        # Dominant frequency
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 0.0
        dominant_power = power[dominant_idx] / (np.sum(power) + 1e-8)

        # Secondary frequency
        power_copy = power.copy()
        power_copy[dominant_idx] = 0
        secondary_idx = np.argmax(power_copy)
        secondary_freq = freqs[secondary_idx] if secondary_idx < len(freqs) else 0.0

        # Speed
        dx = np.diff(cx_arr)
        dy = np.diff(cy_arr)
        speeds = np.sqrt(dx**2 + dy**2)

        # Compile 8-dim gait vector
        sig = np.array([
            dominant_freq,                           # [0] stride rate
            dominant_power,                          # [1] periodicity strength
            secondary_freq,                          # [2] second harmonic
            np.std(h_arr) / (np.mean(h_arr) + 1e-8), # [3] height variance (normalized)
            np.std(w_arr) / (np.mean(w_arr) + 1e-8), # [4] width variance (normalized)
            np.mean(speeds),                         # [5] average speed
            np.std(speeds) / (np.mean(speeds) + 1e-8), # [6] speed consistency
            np.mean(w_arr / (h_arr + 1e-8)),         # [7] aspect ratio
        ], dtype=np.float32)

        self._signatures[track_id] = sig

    def get_signature(self, track_id):
        """
        Get the current gait signature for a track.

        Returns:
            8-dim numpy array, or None if not enough data yet.
        """
        return self._signatures.get(track_id)

    def save_signature(self, track_id):
        """
        Save and return signature for a track being lost (for warm memory).

        Returns:
            8-dim numpy array or None.
        """
        return self._signatures.get(track_id)

    @staticmethod
    def compare(sig_a, sig_b):
        """
        Compare two gait signatures.

        Uses weighted cosine similarity with emphasis on frequency features.

        Args:
            sig_a: 8-dim gait vector.
            sig_b: 8-dim gait vector.

        Returns:
            float: Similarity score 0.0 to 1.0.
        """
        if sig_a is None or sig_b is None:
            return 0.5  # Neutral when gait data unavailable

        # Weights: frequency features are most discriminative
        weights = np.array([3.0, 2.0, 1.5, 1.0, 1.0, 1.5, 1.0, 0.5],
                           dtype=np.float32)

        wa = sig_a * weights
        wb = sig_b * weights

        dot = np.dot(wa, wb)
        norm_a = np.linalg.norm(wa)
        norm_b = np.linalg.norm(wb)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.5

        return float(max(0.0, min(1.0, dot / (norm_a * norm_b))))

    def clear_track(self, track_id):
        """Clean up when a track is removed."""
        self._windows.pop(track_id, None)
        self._signatures.pop(track_id, None)
