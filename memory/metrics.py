"""
SELFWATCH — Tracking Evaluation Metrics

Runtime metrics accumulator for scientific evaluation of tracking quality.
Tracks ID switches, resurrection accuracy, memory survival, and overall
tracking continuity. Exportable to CSV for research analysis.
"""

import time
import csv
import os


class TrackingMetrics:
    """
    Accumulates tracking quality metrics during runtime.

    All methods are O(1) except export_csv and print_summary.
    Zero overhead when not recording (just counter increments).
    """

    def __init__(self):
        self._start_time = time.time()
        self._total_frames = 0

        # ── Counters ─────────────────────────────────────────────────
        self.id_switches = 0
        self.resurrections = 0
        self.false_resurrections = 0
        self.retrieval_attempts = 0
        self.retrieval_successes = 0
        self.memory_saves = 0
        self.memory_decays = 0
        self.memory_prunes = 0
        self.lock_events = 0
        self.soft_locks = 0
        self.hard_locks = 0
        
        # Lock reasons
        self.lock_reasons = {
            "frame_edge": 0,
            "aspect_shift": 0,
            "area_drop": 0,
            "low_confidence": 0,
            "other": 0
        }

        # ── Visual identity continuity metrics ────────────────────────
        # These track the HUMAN-PERCEIVED tracking quality,
        # not just technical ID switch counts.
        self.visible_id_changes = 0      # rendered ID changes in same spatial region
        self.duplicate_box_frames = 0    # frames with 2+ overlapping rendered boxes
        self.fragmentation_count = 0     # track dies and respawns nearby
        self._stability_history = []     # rolling window for stability score

        # ── Timeseries data ──────────────────────────────────────────
        self._identity_lifetimes = []       # (track_id, frames)
        self._memory_survival_times = []    # (track_id, seconds)
        self._resurrection_sims = []        # (track_id, similarity, gap_s)

    def tick_frame(self):
        """Call once per frame to track total processed frames."""
        self._total_frames += 1

    # ── Recording methods ────────────────────────────────────────────

    def record_resurrection(self, track_id, similarity, gap_seconds):
        """Record a successful identity resurrection."""
        self.resurrections += 1
        self._resurrection_sims.append(
            (int(track_id), round(similarity, 4), round(gap_seconds, 2))
        )

    def record_false_resurrection(self, old_id, new_id):
        """Record a false resurrection (manual annotation required)."""
        self.false_resurrections += 1

    def record_id_switch(self, old_id, new_id):
        """Record an ID switch event."""
        self.id_switches += 1

    def record_retrieval_attempt(self, success):
        """Record a retrieval attempt (whether it succeeded or not)."""
        self.retrieval_attempts += 1
        if success:
            self.retrieval_successes += 1

    def record_memory_save(self, track_id, lifetime_frames):
        """Record a track being saved to warm memory."""
        self.memory_saves += 1
        self._identity_lifetimes.append(
            (int(track_id), int(lifetime_frames))
        )

    def record_memory_decay(self, track_id, survival_seconds):
        """Record a track expiring from warm memory."""
        self.memory_decays += 1
        self._memory_survival_times.append(
            (int(track_id), round(survival_seconds, 2))
        )

    def record_memory_prune(self):
        """Record a memory entry being pruned for capacity."""
        self.memory_prunes += 1

    def record_duplicate_boxes(self, count):
        """Record frames where multiple rendered boxes overlap."""
        if count > 0:
            self.duplicate_box_frames += 1

    def record_visible_id_change(self):
        """Record a visible ID changing in the same spatial region."""
        self.visible_id_changes += 1

    def record_fragmentation(self):
        """Record a track dying and respawning nearby."""
        self.fragmentation_count += 1

    def update_stability(self, n_active, n_thinking, n_duplicates):
        """Update rolling identity stability score."""
        # Perfect stability: all active, no thinking, no duplicates
        instability = (n_thinking * 0.1 + n_duplicates * 0.3)
        score = max(0.0, 1.0 - instability / max(n_active, 1))
        self._stability_history.append(score)
        if len(self._stability_history) > 60:
            self._stability_history.pop(0)

    @property
    def identity_stability_score(self):
        """Rolling average identity stability (1.0 = perfect)."""
        if not self._stability_history:
            return 1.0
        return sum(self._stability_history) / len(self._stability_history)

    def record_lock_event(self, reasons=None, lock_type="hard_lock"):
        """Record when identity embedding updates are frozen/penalized."""
        self.lock_events += 1
        if lock_type == "soft_lock":
            self.soft_locks += 1
        else:
            self.hard_locks += 1
            
        if reasons:
            for r in reasons:
                if r in self.lock_reasons:
                    self.lock_reasons[r] += 1
                else:
                    self.lock_reasons["other"] += 1

    # ── Summary ──────────────────────────────────────────────────────

    def get_summary(self):
        """Compute and return all metrics as a dictionary."""
        runtime = time.time() - self._start_time
        lifetimes = [lt for _, lt in self._identity_lifetimes]
        survivals = [st for _, st in self._memory_survival_times]

        return {
            "runtime_seconds": round(runtime, 1),
            "total_frames": self._total_frames,
            "avg_fps": round(self._total_frames / max(runtime, 0.001), 1),
            "id_switches": self.id_switches,
            "resurrections": self.resurrections,
            "false_resurrections": self.false_resurrections,
            "resurrection_accuracy": (
                round(self.resurrections /
                      max(self.retrieval_attempts, 1), 3)
            ),
            "retrieval_attempts": self.retrieval_attempts,
            "retrieval_success_rate": (
                round(self.retrieval_successes /
                      max(self.retrieval_attempts, 1), 3)
            ),
            "memory_saves": self.memory_saves,
            "memory_decays": self.memory_decays,
            "memory_prunes": self.memory_prunes,
            "lock_events": self.lock_events,
            "avg_identity_lifetime": (
                round(sum(lifetimes) / max(len(lifetimes), 1), 1)
            ),
            "avg_memory_survival": (
                round(sum(survivals) / max(len(survivals), 1), 2)
            ),
            "tracking_continuity": (
                round(1.0 - self.id_switches /
                      max(self._total_frames, 1), 4)
            ),
            # Visual identity metrics
            "visible_id_changes": self.visible_id_changes,
            "duplicate_box_frames": self.duplicate_box_frames,
            "fragmentation_count": self.fragmentation_count,
            "identity_stability": round(self.identity_stability_score, 3),
        }

    def print_summary(self):
        """Pretty-print metrics summary to terminal."""
        s = self.get_summary()
        print("\n" + "=" * 60)
        print(f"  SELFWATCH - Cognitive Memory Metrics Summary")
        print("=" * 60)
        print(f"  Runtime           : {s['runtime_seconds']:.1f}s "
              f"({s['total_frames']} frames, {s['avg_fps']} avg FPS)")
        print(f"  " + "-" * 46)
        print(f"  ID Switches       : {s['id_switches']}")
        print(f"  Resurrections     : {s['resurrections']}")
        print(f"  False Resurrections: {s['false_resurrections']}")
        print(f"  Resurrection Acc  : {s['resurrection_accuracy']:.1%}")
        print(f"  " + "-" * 46)
        print(f"  Retrieval Attempts: {s['retrieval_attempts']}")
        print(f"  Retrieval Success : {s['retrieval_success_rate']:.1%}")
        print(f"  " + "-" * 46)
        print(f"  Memory Saves      : {s['memory_saves']}")
        print(f"  Memory Decays     : {s['memory_decays']}")
        print(f"  Memory Prunes     : {s['memory_prunes']}")
        print(f"  Lock Events       : {self.lock_events} (Hard: {self.hard_locks}, Soft: {self.soft_locks})")
        print(f"  Lock Reasons      : Edge={self.lock_reasons['frame_edge']} | "
              f"Aspect={self.lock_reasons['aspect_shift']} | "
              f"Area={self.lock_reasons['area_drop']} | "
              f"Conf={self.lock_reasons['low_confidence']}")
        print( "  ----------------------------------------------")
        print(f"  Avg ID Lifetime   : {s['avg_identity_lifetime']:.0f} frames")
        print(f"  Avg Memory Surv.  : {s['avg_memory_survival']:.1f}s")
        print(f"  Tracking Continuity: {s['tracking_continuity']:.2%}")
        print(f"  " + "-" * 46)
        print(f"  Visible ID Changes : {s['visible_id_changes']}")
        print(f"  Duplicate Box Frames: {s['duplicate_box_frames']}")
        print(f"  Fragmentation      : {s['fragmentation_count']}")
        print(f"  Identity Stability : {s['identity_stability']:.1%}")
        print("=" * 60)

    def export_csv(self, path="logs/metrics_summary.csv"):
        """Export metrics summary to CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        s = self.get_summary()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in s.items():
                writer.writerow([k, v])
