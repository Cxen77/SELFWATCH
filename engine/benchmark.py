"""
SELFWATCH — Benchmark Engine

Runs evaluation sequences through the SELFWATCH pipeline, collecting
comprehensive tracking metrics and failure analysis data. Operates
asynchronously so the UI remains responsive during benchmarking.

Metrics collected:
  - ID Switches, Fragmentations, Recovery events
  - C-BIoU recoveries, Suppression events
  - Average track lifetime, Identity continuity
  - Failure taxonomy classification
  - Per-frame FPS profiling
"""

import os
import cv2
import csv
import json
import time
import threading
import numpy as np
from collections import defaultdict


class BenchmarkResult:
    """Stores results for a single benchmark sequence."""

    def __init__(self, sequence_name, dataset_name):
        self.sequence_name = sequence_name
        self.dataset_name = dataset_name
        self.start_time = time.time()
        self.end_time = None

        # ── Core Tracking Metrics ────────────────────────────────────
        self.total_frames = 0
        self.id_switches = 0
        self.fragmentations = 0
        self.recoveries = 0
        self.cbiou_recoveries = 0
        self.suppression_events = 0
        self.avg_track_lifetime = 0.0
        self.identity_continuity = 0.0

        # ── Failure Taxonomy ─────────────────────────────────────────
        self.failure_counts = {
            "crossing": 0,
            "occlusion": 0,
            "fragmentation": 0,
            "ambiguity": 0,
            "suppression_expiry": 0,
        }

        # ── Per-Frame Timeseries ─────────────────────────────────────
        self.fps_history = []            # per-frame FPS
        self.active_tracks_history = []  # active tracks per frame
        self.switch_frames = []          # frame indices where ID switch occurred
        self.fragmentation_frames = []   # frame indices where fragmentation occurred

        # ── Track Lifetimes ──────────────────────────────────────────
        self.track_lifetimes = {}        # track_id -> lifetime in frames
        self.track_birth_frame = {}      # track_id -> first frame seen

    def finalize(self, pipeline_metrics):
        """Pull final stats from the pipeline metrics object."""
        self.end_time = time.time()

        if pipeline_metrics:
            summary = pipeline_metrics.get_summary()
            self.id_switches = summary.get("id_switches", 0)
            self.recoveries = summary.get("resurrections", 0)
            self.avg_track_lifetime = summary.get("avg_identity_lifetime", 0)
            self.identity_continuity = summary.get("tracking_continuity", 0)

    @property
    def runtime_seconds(self):
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)

    @property
    def avg_fps(self):
        if not self.fps_history:
            return 0.0
        return round(sum(self.fps_history) / len(self.fps_history), 1)

    def to_dict(self):
        """Full result as serializable dictionary."""
        return {
            "sequence": self.sequence_name,
            "dataset": self.dataset_name,
            "runtime_s": self.runtime_seconds,
            "total_frames": self.total_frames,
            "avg_fps": self.avg_fps,
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "recoveries": self.recoveries,
            "cbiou_recoveries": self.cbiou_recoveries,
            "suppression_events": self.suppression_events,
            "avg_track_lifetime": self.avg_track_lifetime,
            "identity_continuity": self.identity_continuity,
            "failure_taxonomy": dict(self.failure_counts),
            "switch_count_by_type": dict(self.failure_counts),
        }


class BenchmarkEngine:
    """
    Orchestrates benchmark runs over dataset sequences.
    
    Usage:
        engine = BenchmarkEngine(pipeline, dataset_manager)
        engine.run_sequence(seq_info, progress_callback, done_callback)
    """

    def __init__(self, pipeline, dataset_manager, forensic_debugger=None):
        self.pipeline = pipeline
        self.dataset_manager = dataset_manager
        self.forensic = forensic_debugger
        self.is_running = False
        self._thread = None
        self.current_result = None

        # Accumulated results across all runs this session
        self.all_results = []

    def run_sequence(self, seq_info, progress_cb=None, done_cb=None,
                     frame_cb=None):
        """
        Run benchmark on a single sequence (async).
        
        Args:
            seq_info: SequenceInfo from DatasetManager
            progress_cb: fn(frame_idx, total_frames, stats_dict)
            done_cb: fn(BenchmarkResult)
            frame_cb: fn(annotated_frame, stats) for live display
        """
        if self.is_running:
            print("[Benchmark] Already running, ignoring request.")
            return

        self.is_running = True
        self._thread = threading.Thread(
            target=self._run_worker,
            args=(seq_info, progress_cb, done_cb, frame_cb),
            daemon=True, name="sw-benchmark"
        )
        self._thread.start()

    def stop(self):
        """Cancel a running benchmark."""
        self.is_running = False

    def _run_worker(self, seq_info, progress_cb, done_cb, frame_cb):
        """Worker thread: iterate frames and process through pipeline."""
        result = BenchmarkResult(seq_info.name, seq_info.dataset)
        self.current_result = result

        print(f"\n[Benchmark] Starting: {seq_info.name} ({seq_info.dataset})")

        # Open source
        cap = None
        if seq_info.is_video:
            cap = cv2.VideoCapture(seq_info.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FPS) * 
                             cap.get(cv2.CAP_PROP_FRAME_COUNT) / 
                             max(cap.get(cv2.CAP_PROP_FPS), 1))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            total_frames = seq_info.seq_length

        # Reset pipeline for clean run
        if hasattr(self.pipeline, 'reset'):
            self.pipeline.reset()

        frame_idx = 0
        last_time = time.perf_counter()

        while self.is_running:
            # Read frame
            frame = None
            if seq_info.is_video:
                if cap is None:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                # Image sequence
                frame_idx_1based = frame_idx + 1
                img_path = os.path.join(
                    seq_info.frame_dir,
                    f"{frame_idx_1based:06d}.jpg"
                )
                if not os.path.isfile(img_path):
                    break
                frame = cv2.imread(img_path)
                if frame is None:
                    break

            frame_idx += 1

            # Process through pipeline
            t0 = time.perf_counter()
            annotated, stats = self.pipeline.process_frame(
                frame, frame_delta=1, frame_index=frame_idx
            )
            t1 = time.perf_counter()

            # Record FPS
            frame_fps = 1.0 / max(t1 - t0, 1e-6)
            result.fps_history.append(frame_fps)
            result.total_frames = frame_idx
            result.active_tracks_history.append(
                stats.get("active_tracks", 0)
            )

            # Detect events from stats
            if stats.get("id_switches"):
                switch_list = stats["id_switches"]
                if isinstance(switch_list, list):
                    for sw in switch_list:
                        result.switch_frames.append(frame_idx)
                        # Classify
                        taxonomy = self._classify_switch(stats)
                        if taxonomy in result.failure_counts:
                            result.failure_counts[taxonomy] += 1
                else:
                    result.switch_frames.append(frame_idx)

            # Count C-BIoU recoveries from stats
            if stats.get("cbiou_matches", 0):
                result.cbiou_recoveries += stats["cbiou_matches"]

            # Count suppression events
            if stats.get("births_suppressed", 0):
                result.suppression_events += stats["births_suppressed"]

            # Count fragmentations (new track births)
            if stats.get("new_tracks_spawned", 0):
                result.fragmentations += stats["new_tracks_spawned"]

            # Forensic capture on ID switch
            if stats.get("id_switches") and self.forensic:
                threading.Thread(
                    target=self.forensic.capture_id_switch,
                    args=(None, stats, annotated),
                    daemon=True
                ).start()

            # Callbacks
            if progress_cb:
                try:
                    progress_cb(frame_idx, total_frames, stats)
                except Exception:
                    pass

            if frame_cb:
                try:
                    frame_cb(annotated, stats)
                except Exception:
                    pass

            last_time = t1

        # Finalize
        if cap:
            cap.release()

        # Pull metrics from pipeline
        pipeline_metrics = getattr(self.pipeline, 'metrics', None)
        result.finalize(pipeline_metrics)

        self.all_results.append(result)
        self.is_running = False
        self.current_result = None

        print(f"[Benchmark] Complete: {seq_info.name}")
        print(f"  Frames: {result.total_frames} | "
              f"Avg FPS: {result.avg_fps} | "
              f"ID Switches: {result.id_switches}")

        if done_cb:
            try:
                done_cb(result)
            except Exception as e:
                print(f"[Benchmark] Done callback error: {e}")

    def _classify_switch(self, stats):
        """Classify an ID switch based on current pipeline state."""
        frozen = stats.get("frozen_gids", [])
        suppress = stats.get("suppress_regions", [])
        thinking = stats.get("thinking_gids", [])

        if frozen:
            return "crossing"
        if suppress:
            return "suppression_expiry"
        if thinking:
            return "occlusion"

        return "fragmentation"

    # ── Export ────────────────────────────────────────────────────────

    def export_results_csv(self, path="logs/benchmark_results.csv"):
        """Export all accumulated results to CSV."""
        if not self.all_results:
            print("[Benchmark] No results to export.")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        fieldnames = [
            "sequence", "dataset", "runtime_s", "total_frames",
            "avg_fps", "id_switches", "fragmentations", "recoveries",
            "cbiou_recoveries", "suppression_events",
            "avg_track_lifetime", "identity_continuity",
            "fail_crossing", "fail_occlusion", "fail_fragmentation",
            "fail_ambiguity", "fail_suppression_expiry",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.all_results:
                row = {
                    "sequence": r.sequence_name,
                    "dataset": r.dataset_name,
                    "runtime_s": r.runtime_seconds,
                    "total_frames": r.total_frames,
                    "avg_fps": r.avg_fps,
                    "id_switches": r.id_switches,
                    "fragmentations": r.fragmentations,
                    "recoveries": r.recoveries,
                    "cbiou_recoveries": r.cbiou_recoveries,
                    "suppression_events": r.suppression_events,
                    "avg_track_lifetime": r.avg_track_lifetime,
                    "identity_continuity": r.identity_continuity,
                    "fail_crossing": r.failure_counts.get("crossing", 0),
                    "fail_occlusion": r.failure_counts.get("occlusion", 0),
                    "fail_fragmentation": r.failure_counts.get("fragmentation", 0),
                    "fail_ambiguity": r.failure_counts.get("ambiguity", 0),
                    "fail_suppression_expiry": r.failure_counts.get("suppression_expiry", 0),
                }
                writer.writerow(row)

        print(f"[Benchmark] Results exported to {path}")

    def export_results_json(self, path="logs/benchmark_results.json"):
        """Export all results as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [r.to_dict() for r in self.all_results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Benchmark] Results exported to {path}")
