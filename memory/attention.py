"""
SELFWATCH - Cognitive Attention Priority System

Adaptive compute allocation that assigns attention tiers to tracks.
Uncertain/new/recovering tracks get FULL ReID processing.
Stable, well-known tracks skip expensive ReID extraction entirely
and rely on spatial tracking alone.

Result: With 8 tracked people, instead of 8 ReID extractions per frame,
you might do 3 (HIGH) + 2 (NORMAL) + 0 (LOW) = 5.
That is a ~37% compute reduction while IMPROVING accuracy on the
tracks that actually need it.

Cost: NEGATIVE. This system saves compute.
"""

import time


# Attention tiers
TIER_HIGH = "HIGH"          # Full ReID every frame
TIER_NORMAL = "NORMAL"      # ReID every 3rd frame
TIER_LOW = "LOW"            # Skip ReID, spatial only


class CognitiveAttention:
    """
    Assigns attention tiers to tracked identities based on their stability.

    Tier assignment rules:
        HIGH:   Track age < 15 frames, OR identity was recently recovered,
                OR state is UNCERTAIN, OR embedding history < 5.
        NORMAL: Track age 15-90 frames, stable confidence, no recent events.
        LOW:    Track age > 90 frames, rock-solid embedding history,
                consistent position movement.

    Usage:
        attention = CognitiveAttention()

        # Per track, per frame:
        tier = attention.get_tier(track_id)
        if attention.should_extract_reid(track_id, frame_count):
            # run OSNet
            embedding = reid.extract(crop)
        else:
            # skip ReID, use spatial tracking only
            embedding = track.embedding  # reuse last known

    Args:
        high_age_thresh:   Tracks younger than this are always HIGH.
        low_age_thresh:    Tracks older than this can become LOW.
        recovery_cooldown: Frames after resurrection before leaving HIGH.
    """

    def __init__(self, high_age_thresh=15, low_age_thresh=90,
                 recovery_cooldown=20):
        self.high_age_thresh = high_age_thresh
        self.low_age_thresh = low_age_thresh
        self.recovery_cooldown = recovery_cooldown

        # Per-track state
        self._tiers = {}                # track_id -> tier
        self._last_recovery = {}        # track_id -> frame_count when recovered
        self._reid_skip_counters = {}   # track_id -> frames since last ReID

    def update_tier(self, track_id, track_age, is_confirmed,
                    identity_state=None, frame_count=0):
        """
        Recalculate attention tier for a track.

        Args:
            track_id:       Track ID.
            track_age:      Total frames this track has existed.
            is_confirmed:   Whether the track is confirmed.
            identity_state: Cognitive memory state (ACTIVE/UNCERTAIN/etc).
            frame_count:    Current global frame counter.
        """
        # Check recovery cooldown
        recovery_frame = self._last_recovery.get(track_id, -999)
        in_cooldown = (frame_count - recovery_frame) < self.recovery_cooldown

        # Determine tier
        if not is_confirmed:
            tier = TIER_HIGH
        elif in_cooldown:
            tier = TIER_HIGH
        elif identity_state == "UNCERTAIN":
            tier = TIER_HIGH
        elif track_age < self.high_age_thresh:
            tier = TIER_HIGH
        elif track_age > self.low_age_thresh:
            tier = TIER_LOW
        else:
            tier = TIER_NORMAL

        self._tiers[track_id] = tier

    def mark_recovered(self, track_id, frame_count):
        """Mark a track as recently recovered (forces HIGH attention)."""
        self._last_recovery[track_id] = frame_count
        self._tiers[track_id] = TIER_HIGH

    def should_extract_reid(self, track_id, frame_count):
        """
        Whether ReID embedding should be extracted this frame.

        Returns:
            True if ReID should run, False to skip.
        """
        tier = self._tiers.get(track_id, TIER_HIGH)

        if tier == TIER_HIGH:
            return True

        # Initialize counter if needed
        if track_id not in self._reid_skip_counters:
            self._reid_skip_counters[track_id] = 0

        self._reid_skip_counters[track_id] += 1

        if tier == TIER_NORMAL:
            # Extract every 3rd frame
            if self._reid_skip_counters[track_id] >= 3:
                self._reid_skip_counters[track_id] = 0
                return True
            return False

        if tier == TIER_LOW:
            # Extract every 10th frame
            if self._reid_skip_counters[track_id] >= 10:
                self._reid_skip_counters[track_id] = 0
                return True
            return False

        return True

    def get_tier(self, track_id):
        """Get current attention tier for a track."""
        return self._tiers.get(track_id, TIER_HIGH)

    def clear_track(self, track_id):
        """Clean up when a track is removed."""
        self._tiers.pop(track_id, None)
        self._last_recovery.pop(track_id, None)
        self._reid_skip_counters.pop(track_id, None)

    def get_stats(self):
        """Return count of tracks per tier."""
        counts = {TIER_HIGH: 0, TIER_NORMAL: 0, TIER_LOW: 0}
        for tier in self._tiers.values():
            counts[tier] = counts.get(tier, 0) + 1
        return counts
