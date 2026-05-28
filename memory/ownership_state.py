"""
Central Ownership State Machine — Single Authority for Identity Ownership

This is the ONLY module that decides identity ownership state.
No other module may independently determine or override ownership.

States:
  VISIBLE_ACTIVE    — Real detection, confirmed, currently rendered
  VISIBLE_FROZEN    — In occlusion group, rendered but ownership locked
  LATENT_CANDIDATE  — Track lost, within cognitive hold window, NOT rendered
  RECOVERING        — Recovery accepted internally, visual hysteresis pending
  ARCHIVED          — Moved to warm/phantom memory, no longer actively tracked

Responsibilities:
  - Canonical state for every global_id
  - State transition logic with guards
  - Forensic logging of all transitions
  - Provides single source of truth for all subsystems

Does NOT do:
  - Rendering decisions (that's the renderer's job)
  - Metric counting (that's the evaluator's job)
  - Spatial tracking (that's the tracker's job)
"""

from collections import defaultdict
import time
import math


# ── State Constants ─────────────────────────────────────────────────
VISIBLE_ACTIVE = 0
VISIBLE_FROZEN = 1
LATENT_CANDIDATE = 2
RECOVERING = 3
ARCHIVED = 4

STATE_NAMES = {
    VISIBLE_ACTIVE: "VISIBLE_ACTIVE",
    VISIBLE_FROZEN: "VISIBLE_FROZEN",
    LATENT_CANDIDATE: "LATENT_CANDIDATE",
    RECOVERING: "RECOVERING",
    ARCHIVED: "ARCHIVED",
}


class IdentityRecord:
    """Canonical ownership record for a single global identity."""

    __slots__ = [
        "gid", "state", "entered_frame", "owning_lid",
        "last_box", "last_velocity", "last_embedding",
        "confidence", "stability_frames", "recovery_frames",
        "freeze_reason", "archive_reason",
    ]

    def __init__(self, gid, state, frame):
        self.gid = gid
        self.state = state
        self.entered_frame = frame
        self.owning_lid = None
        self.last_box = None
        self.last_velocity = None
        self.last_embedding = None
        self.confidence = 0.5
        self.stability_frames = 0
        self.recovery_frames = 0
        self.freeze_reason = None
        self.archive_reason = None


class OwnershipStateMachine:
    """
    Central authority for identity ownership.

    All ownership queries and state changes go through this single module.
    Other subsystems (tracker, memory, renderer) query state but never set it.
    """

    # ── Configuration ────────────────────────────────────────────────
    RECOVERY_HYSTERESIS = 8     # Frames of consistent recovery before VISIBLE
    CONFIDENCE_GROW_RATE = 0.05  # Per-frame confidence growth when active
    CONFIDENCE_DECAY_RATE = 0.03  # Per-frame confidence decay when latent
    MAX_TRANSITION_LOG = 200     # Recent transitions to keep for forensics

    def __init__(self):
        # Canonical state store: gid -> IdentityRecord
        self._records = {}

        # Transition log for forensic analysis
        self._transition_log = []

        self._frame_count = 0

    # ── State Query API ──────────────────────────────────────────────

    def get_state(self, gid):
        """Get the canonical ownership state for a global identity."""
        record = self._records.get(gid)
        return record.state if record else None

    def get_record(self, gid):
        """Get the full ownership record for a global identity."""
        return self._records.get(gid)

    def get_state_name(self, gid):
        """Get human-readable state name."""
        state = self.get_state(gid)
        return STATE_NAMES.get(state, "UNKNOWN") if state is not None else "NONE"

    def get_all_in_state(self, state):
        """Return dict of gid -> IdentityRecord for all identities in a given state."""
        return {gid: rec for gid, rec in self._records.items()
                if rec.state == state}

    def get_visible_gids(self):
        """Return set of gids that should be visually rendered."""
        return {gid for gid, rec in self._records.items()
                if rec.state in (VISIBLE_ACTIVE, VISIBLE_FROZEN)}

    def get_latent_gids(self):
        """Return set of gids in non-visible states (candidate/recovering)."""
        return {gid for gid, rec in self._records.items()
                if rec.state in (LATENT_CANDIDATE, RECOVERING)}

    def is_visible(self, gid):
        """Check if a gid should be rendered."""
        record = self._records.get(gid)
        return record is not None and record.state in (VISIBLE_ACTIVE, VISIBLE_FROZEN)

    def is_frozen(self, gid):
        """Check if a gid is ownership-locked (frozen)."""
        record = self._records.get(gid)
        return record is not None and record.state == VISIBLE_FROZEN

    def get_confidence(self, gid):
        """Get ownership confidence [0.0, 1.0]."""
        record = self._records.get(gid)
        return record.confidence if record else 0.0

    @property
    def active_count(self):
        return sum(1 for r in self._records.values()
                   if r.state == VISIBLE_ACTIVE)

    @property
    def frozen_count(self):
        return sum(1 for r in self._records.values()
                   if r.state == VISIBLE_FROZEN)

    @property
    def latent_count(self):
        return sum(1 for r in self._records.values()
                   if r.state in (LATENT_CANDIDATE, RECOVERING))

    # ── State Transition API ─────────────────────────────────────────

    def tick(self, frame_count):
        """Advance frame counter. Call once per frame before transitions."""
        self._frame_count = frame_count

        # Update confidence for all records
        for gid, rec in self._records.items():
            if rec.state == VISIBLE_ACTIVE:
                rec.confidence = min(1.0, rec.confidence + self.CONFIDENCE_GROW_RATE)
                rec.stability_frames += 1
            elif rec.state == VISIBLE_FROZEN:
                # Frozen: confidence stable (no grow, no decay)
                rec.stability_frames += 1
            elif rec.state == LATENT_CANDIDATE:
                rec.confidence = max(0.0, rec.confidence - self.CONFIDENCE_DECAY_RATE)
                rec.stability_frames = 0
            elif rec.state == RECOVERING:
                rec.recovery_frames += 1

    def activate(self, gid, owning_lid=None, box=None, velocity=None):
        """
        Transition to VISIBLE_ACTIVE.

        Valid from: NEW, LATENT_CANDIDATE, RECOVERING, VISIBLE_FROZEN
        """
        record = self._records.get(gid)
        old_state = record.state if record else None

        if record is None:
            record = IdentityRecord(gid, VISIBLE_ACTIVE, self._frame_count)
            self._records[gid] = record
        else:
            record.state = VISIBLE_ACTIVE
            record.entered_frame = self._frame_count
            record.freeze_reason = None

        record.owning_lid = owning_lid
        if box is not None:
            record.last_box = box
        if velocity is not None:
            record.last_velocity = velocity
        record.recovery_frames = 0

        self._log_transition(gid, old_state, VISIBLE_ACTIVE, "activate")

    def freeze(self, gid, reason="overlap"):
        """
        Transition to VISIBLE_FROZEN.

        Valid from: VISIBLE_ACTIVE
        """
        record = self._records.get(gid)
        if record is None or record.state != VISIBLE_ACTIVE:
            return  # Can only freeze active identities

        old_state = record.state
        record.state = VISIBLE_FROZEN
        record.freeze_reason = reason
        record.entered_frame = self._frame_count

        self._log_transition(gid, old_state, VISIBLE_FROZEN, f"freeze:{reason}")

    def unfreeze(self, gid):
        """
        Transition from VISIBLE_FROZEN back to VISIBLE_ACTIVE.

        Valid from: VISIBLE_FROZEN (after overlap resolved)
        """
        record = self._records.get(gid)
        if record is None or record.state != VISIBLE_FROZEN:
            return

        old_state = record.state
        record.state = VISIBLE_ACTIVE
        record.freeze_reason = None
        record.entered_frame = self._frame_count

        self._log_transition(gid, old_state, VISIBLE_ACTIVE, "unfreeze")

    def to_latent(self, gid, reason="lost"):
        """
        Transition to LATENT_CANDIDATE (track lost, within hold window).

        Valid from: VISIBLE_ACTIVE, VISIBLE_FROZEN
        """
        record = self._records.get(gid)
        if record is None:
            return
        if record.state not in (VISIBLE_ACTIVE, VISIBLE_FROZEN):
            return  # Already latent or archived

        old_state = record.state
        record.state = LATENT_CANDIDATE
        record.entered_frame = self._frame_count
        record.recovery_frames = 0

        self._log_transition(gid, old_state, LATENT_CANDIDATE, f"to_latent:{reason}")

    def begin_recovery(self, gid, proposed_lid=None):
        """
        Transition to RECOVERING (recovery accepted, hysteresis pending).

        Valid from: LATENT_CANDIDATE
        """
        record = self._records.get(gid)
        if record is None or record.state != LATENT_CANDIDATE:
            return False

        old_state = record.state
        record.state = RECOVERING
        record.entered_frame = self._frame_count
        record.recovery_frames = 0
        if proposed_lid is not None:
            record.owning_lid = proposed_lid

        self._log_transition(gid, old_state, RECOVERING, "begin_recovery")
        return True

    def complete_recovery(self, gid):
        """
        Complete recovery: RECOVERING -> VISIBLE_ACTIVE.

        Only succeeds after RECOVERY_HYSTERESIS frames.
        """
        record = self._records.get(gid)
        if record is None or record.state != RECOVERING:
            return False

        if record.recovery_frames < self.RECOVERY_HYSTERESIS:
            return False  # Not enough consistent frames yet

        old_state = record.state
        record.state = VISIBLE_ACTIVE
        record.entered_frame = self._frame_count
        record.recovery_frames = 0
        record.confidence = 0.5  # Reset confidence for recovered identity

        self._log_transition(gid, old_state, VISIBLE_ACTIVE, "recovery_complete")
        return True

    def cancel_recovery(self, gid, reason="inconsistent"):
        """
        Cancel recovery: RECOVERING -> LATENT_CANDIDATE.

        Called when recovery evidence becomes inconsistent.
        """
        record = self._records.get(gid)
        if record is None or record.state != RECOVERING:
            return

        old_state = record.state
        record.state = LATENT_CANDIDATE
        record.entered_frame = self._frame_count
        record.recovery_frames = 0

        self._log_transition(gid, old_state, LATENT_CANDIDATE,
                             f"cancel_recovery:{reason}")

    def archive(self, gid, reason="expired"):
        """
        Transition to ARCHIVED (moved to warm memory).

        Valid from: LATENT_CANDIDATE, RECOVERING
        """
        record = self._records.get(gid)
        if record is None:
            return
        if record.state in (VISIBLE_ACTIVE, VISIBLE_FROZEN):
            return  # Cannot archive visible identities directly

        old_state = record.state
        record.state = ARCHIVED
        record.archive_reason = reason
        record.entered_frame = self._frame_count

        self._log_transition(gid, old_state, ARCHIVED, f"archive:{reason}")

    def remove(self, gid):
        """Remove an identity record entirely (after archival processing)."""
        if gid in self._records:
            del self._records[gid]

    # ── Bulk Operations ──────────────────────────────────────────────

    def update_frozen_set(self, frozen_gids):
        """
        Sync frozen state with occlusion manager output.

        Identities in frozen_gids become VISIBLE_FROZEN.
        Identities NOT in frozen_gids that were VISIBLE_FROZEN become VISIBLE_ACTIVE.
        """
        for gid, rec in self._records.items():
            if gid in frozen_gids:
                if rec.state == VISIBLE_ACTIVE:
                    self.freeze(gid, reason="overlap")
            else:
                if rec.state == VISIBLE_FROZEN:
                    self.unfreeze(gid)

    # ── Forensic Logging ─────────────────────────────────────────────

    def _log_transition(self, gid, old_state, new_state, reason):
        """Log every state transition for forensic analysis."""
        old_name = STATE_NAMES.get(old_state, "NONE")
        new_name = STATE_NAMES.get(new_state, "?")

        entry = {
            "frame": self._frame_count,
            "gid": gid,
            "from": old_name,
            "to": new_name,
            "reason": reason,
            "timestamp": time.perf_counter(),
        }
        self._transition_log.append(entry)
        if len(self._transition_log) > self.MAX_TRANSITION_LOG:
            self._transition_log.pop(0)

    def get_transition_log(self, last_n=50):
        """Return recent transitions for forensic inspection."""
        return list(self._transition_log[-last_n:])

    def get_summary(self):
        """Return summary counts by state."""
        counts = defaultdict(int)
        for rec in self._records.values():
            counts[STATE_NAMES.get(rec.state, "UNKNOWN")] += 1
        return dict(counts)
