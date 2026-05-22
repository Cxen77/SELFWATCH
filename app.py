"""
SELFWATCH Testing UI — Deterministic Frame-Owned Pipeline

Architecture (simplest correct design):
  1. READER thread  — reads frames, keeps ONLY the latest (buffer=1).
  2. INFERENCE thread — grabs latest frame, runs full pipeline on it.
                        Pipeline draws boxes ON that frame. Publishes
                        the completed annotated frame.
  3. MAIN (UI) thread — displays the latest completed annotated frame.

Zero interpolation. Zero extrapolation. Zero velocity math.
Boxes are ALWAYS on the exact frame they were detected on.
Display updates at inference FPS (~3-5fps). Slight stutter but
perfect temporal correctness and box alignment.
"""

import os
import time
import cv2
import threading
import collections
import numpy as np
import customtkinter as ctk
from tkinter import filedialog

import config
from ui.video_player import VideoPlayer
from ui.side_panel import SidePanel
from ui.timeline import Timeline
from ui.overlay_panel import OverlayPanel
from ui.benchmark_panel import BenchmarkPanel
from engine.state_cache import StateCache
from engine.dataset_manager import DatasetManager
from engine.benchmark import BenchmarkEngine
from memory.forensic_debug import ForensicDebugger
from evaluation.evaluator import SELFWATCHEvaluator
from evaluation.experiment_logs.experiment_tracker import ResearchExperimentTracker

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SelfWatchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SELFWATCH — Cognitive Tracking Lab")
        self.geometry("1400x900")

        # ── Core State ──────────────────────────────────────────────
        self.pipeline = None
        self.cap = None
        self.state_cache = StateCache(max_frames=2000)
        self.forensic = ForensicDebugger()
        self.dataset_manager = DatasetManager(root="datasets")
        self.benchmark_engine = None
        self._sequence_map = {}  # name -> SequenceInfo
        self.is_running = False
        self.is_playing = True
        self.is_video_source = False
        self.video_fps = 30.0
        self.evaluator = None
        self.experiment_tracker = None

        # Thread handles
        self._reader_thread = None
        self._inference_thread = None
        self._display_timer = None

        # ── Reader → Inference: single-slot buffer ──────────────────
        # Only the LATEST frame matters. No ring buffer, no backlog.
        self._frame_slot = None          # (frame_idx, frame) or None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        # ── Inference → Display: completed annotated frame ──────────
        self._display_lock = threading.Lock()
        self._display_frame = None       # latest pipeline-annotated frame
        self._display_stats = None       # latest stats dict
        self._cached_display = None      # held frame for redisplay between ticks

        # ── Counters ────────────────────────────────────────────────
        self._raw_frame_index = 0
        self._frames_read = 0
        self._frames_processed = 0
        self._last_display_raw_frame_index = 0
        self._last_tracker_raw_frame_index = 0
        self._inference_fps = 0.0
        self._last_display_time = time.perf_counter()

        self._build_ui()

    # ════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)  # Right panel
        self.grid_rowconfigure(0, weight=1)

        callbacks = {
            "on_source_camera": self.start_camera,
            "on_source_video": self.start_video,
            "on_toggle_debug": self.toggle_debug,
            "on_play": self.play,
            "on_pause": self.pause,
            "on_next_frame": self.step_forward,
            "on_prev_frame": self.step_backward,
            "on_scrub": self.scrub_to,
            "on_toggle_layer": self.toggle_layer,
        }

        benchmark_callbacks = {
            "on_benchmark_toggle": self._on_benchmark_toggle,
            "on_dataset_change": self._on_dataset_change,
            "on_benchmark_run": self._on_benchmark_run,
            "on_benchmark_stop": self._on_benchmark_stop,
            "on_benchmark_export": self._on_benchmark_export,
        }

        # Side Panel (Left) — uses scrollable frame for long content
        self.side_panel_scroll = ctk.CTkScrollableFrame(
            self, width=300, fg_color=("gray92", "gray14")
        )
        self.side_panel_scroll.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.side_panel = SidePanel(
            self.side_panel_scroll, callbacks,
            fg_color="transparent"
        )
        self.side_panel.pack(fill="x", expand=False)

        # Benchmark Panel (inside left scroll)
        self.benchmark_panel = BenchmarkPanel(
            self.side_panel_scroll, benchmark_callbacks,
            fg_color="transparent"
        )
        self.benchmark_panel.pack(fill="x", expand=False, pady=(5, 0))

        # Video Player (Center)
        self.video_player = VideoPlayer(self)
        self.video_player.grid(row=0, column=1, sticky="nsew", padx=10, pady=(10, 0))

        # Timeline (Bottom)
        self.timeline = Timeline(self, callbacks, height=80)
        self.timeline.grid(row=1, column=1, sticky="ew", padx=10, pady=10)

        # Overlay Panel (Right)
        self.overlay_panel = OverlayPanel(self, callbacks, width=250)
        self.overlay_panel.grid(row=0, column=2, rowspan=2, sticky="nsew")

        # Key bindings
        self.bind("1", lambda e: self.overlay_panel.toggle_master_checkbox("tracking"))
        self.bind("2", lambda e: self.overlay_panel.toggle_master_checkbox("motion"))
        self.bind("3", lambda e: self.overlay_panel.toggle_master_checkbox("association"))
        self.bind("4", lambda e: self.overlay_panel.toggle_master_checkbox("cognitive"))
        self.bind("5", lambda e: self.overlay_panel.toggle_master_checkbox("forensic"))

    # ════════════════════════════════════════════════════════════════
    #  ENGINE INITIALIZATION (lazy)
    # ════════════════════════════════════════════════════════════════

    def init_pipeline(self):
        if self.pipeline is not None:
            return

        # Read model variant from UI dropdown
        variant_map = {
            "RT-DETR-Medium": ("medium", 512),
            "RT-DETR-Large": ("large", 704),
            "RT-DETR-Nano": ("nano", 384),
        }
        selected = self.side_panel.model_var.get()
        variant, resolution = variant_map.get(selected, ("medium", 576))
        print(f"[UI] Loading {selected} (res={resolution})...")

        from detectors import RTDETRDetector
        from reid import EmbeddingExtractor
        from trackers import StrongSORTTracker
        from engine.pipeline import SelfWatchPipeline

        detector = RTDETRDetector(
            variant=variant, resolution=resolution,
            use_amp=True, compile_model=True,
        )
        detector.warmup()

        reid = EmbeddingExtractor(
            weights_path=config.REID_WEIGHTS,
            device=detector.get_device(),
            half=config.REID_HALF,
        )

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

        self.pipeline = SelfWatchPipeline(
            detector, reid, tracker,
            enable_debug_overlay=self.overlay_panel.chk_debug.get(),
        )
        print(f"[UI] Engine loaded ({selected}).")

    # ════════════════════════════════════════════════════════════════
    #  SOURCE CONTROL
    # ════════════════════════════════════════════════════════════════

    def start_camera(self):
        self.is_video_source = False
        self._start_stream_async(0)

    def start_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm")],
        )
        if path:
            self.is_video_source = True
            self._start_stream_async(path)

    def _start_stream_async(self, source):
        if self.is_running:
            self.stop_stream()

        self.is_running = True
        self.video_player.show_loading("Initializing Pipeline...")
        threading.Thread(
            target=self._init_stream_worker, args=(source,),
            daemon=True, name="sw-startup"
        ).start()

    def _init_stream_worker(self, source):
        """Background thread: load models + open video + process first frame."""
        t0 = time.perf_counter()
        if self.pipeline is None:
            self.after(0, self.video_player.show_loading, "Loading Neural Networks...")
            self.init_pipeline()
        print(f"[Profiling] Model load time: {time.perf_counter() - t0:.2f}s")

        self.after(0, self.video_player.show_loading, "Opening Video Source...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[UI] Error: cannot open {source}")
            self.after(0, self.video_player.show_loading, "Failed to open source")
            self.is_running = False
            return

        # ── Process first frame synchronously ─────────────────────
        # Guarantees boxes are ready before display starts.
        self.after(0, self.video_player.show_loading, "Processing first frame...")
        ret, first_frame = cap.read()
        first_annotated = None
        first_stats = None
        if ret:
            first_annotated, first_stats = self.pipeline.process_frame(
                first_frame, frame_delta=1, frame_index=1)
            n = first_stats.get("active_tracks", 0)
            print(f"[Startup] First frame processed: {n} tracks")

        self.after(0, self._finalize_stream_start, cap, source,
                   first_annotated, first_stats)

    def _finalize_stream_start(self, cap, source,
                               first_annotated=None, first_stats=None):
        """UI thread: seed display + start all threads."""
        if not self.is_running:
            cap.release()
            return

        self.state_cache.clear()
        self.cap = cap
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Reset counters (first frame already consumed)
        has_first = first_annotated is not None
        self._raw_frame_index = 1 if has_first else 0
        self._frames_read = 1 if has_first else 0
        self._frames_processed = 1 if has_first else 0
        self._last_display_raw_frame_index = 1 if has_first else 0
        self._last_tracker_raw_frame_index = 1 if has_first else 0

        print("[EVAL INIT] Initializing SELFWATCHEvaluator and ExperimentTracker")
        self.evaluator = SELFWATCHEvaluator()
        self.experiment_tracker = ResearchExperimentTracker()
        
        config_data = {k: v for k, v in vars(config).items() if k.isupper()}
        self.experiment_tracker.save_config(config_data)

        # Seed display with first processed frame
        self._display_frame = first_annotated
        self._display_stats = first_stats
        self._frame_slot = None

        if has_first:
            self.state_cache.append(first_annotated, first_stats)

        self.is_playing = True
        self.timeline.is_playing = True
        self.timeline.btn_play_pause.configure(text="⏸ Pause")

        # Launch threads
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="sw-reader")
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="sw-inference")
        self._reader_thread.start()
        self._inference_thread.start()

        # Start display timer
        if self._display_timer is None:
            self._schedule_display()

    def stop_stream(self):
        self.is_running = False
        self._frame_event.set()

        if self._reader_thread:
            self._reader_thread.join(timeout=0.5)
        if self._inference_thread:
            self._inference_thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.evaluator and self.experiment_tracker:
            print("[EVAL SAVE START] Saving metrics and summary...")
            report = self.evaluator.get_final_report()
            self.experiment_tracker.save_metrics(report)
            avg_fps = self._inference_fps if self._inference_fps else 0.0
            total_time = self._frames_processed / avg_fps if avg_fps > 0 else 0.0
            self.experiment_tracker.log_runtime_stats(avg_fps, total_time)
            import os
            abs_path = os.path.abspath(self.experiment_tracker.log_dir)
            print(f"[EVAL SAVE COMPLETE] Directory: {abs_path}")

    def toggle_debug(self, state):
        if self.pipeline:
            self.pipeline.debug_overlay.enabled = state

    def toggle_layer(self, layer_name, state):
        if self.pipeline and hasattr(self.pipeline, "layer_manager"):
            self.pipeline.layer_manager.toggle(layer_name, state)
            # Force redraw of cached frame if paused
            if not self.is_playing and self._display_frame is not None:
                self.scrub_to(self.timeline.slider_var.get())

    # ════════════════════════════════════════════════════════════════
    #  PLAYBACK & SCRUBBING CONTROL
    # ════════════════════════════════════════════════════════════════

    def play(self):
        self.is_playing = True

    def pause(self):
        self.is_playing = False

    def step_forward(self):
        """Advance exactly one frame through the pipeline (paused mode)."""
        if not self.is_running or self.cap is None:
            return
        self.pause()
        
        # If we are scrubbed backwards, just step forward in the cache
        if self.state_cache.current_index < self.state_cache.total_frames - 1:
            self.state_cache.current_index += 1
            frame, meta = self.state_cache.get_frame(self.state_cache.current_index)
            if frame is not None:
                self.video_player.update_frame(frame)
                self.side_panel.update_metrics(meta)
                self.timeline.update_state(
                    self.state_cache.current_index,
                    max(0, self.state_cache.total_frames - 1),
                    False)
            return

        # Otherwise process a new frame from the live video
        ret, frame = self.cap.read()
        if not ret:
            return
        self._raw_frame_index += 1
        frame_delta = max(1, self._raw_frame_index - self._last_tracker_raw_frame_index)
        annotated, stats = self.pipeline.process_frame(
            frame, frame_delta=frame_delta, frame_index=self._raw_frame_index)
        self._last_display_raw_frame_index = self._raw_frame_index
        self._last_tracker_raw_frame_index = self._raw_frame_index
        stats["inference_fps"] = self._inference_fps
        self.state_cache.append(annotated, stats)
        self.video_player.update_frame(annotated)
        self.side_panel.update_metrics(stats)
        self.timeline.update_state(
            self.state_cache.current_index,
            max(0, self.state_cache.total_frames - 1),
            False)

    def step_backward(self):
        self.pause()
        if self.state_cache.current_index > 0:
            self.state_cache.current_index -= 1
            frame, meta = self.state_cache.get_frame(self.state_cache.current_index)
            if frame is not None:
                self.video_player.update_frame(frame)
                self.side_panel.update_metrics(meta)
                self.timeline.update_state(
                    self.state_cache.current_index,
                    max(0, self.state_cache.total_frames - 1),
                    False)

    def scrub_to(self, index):
        self.pause()
        if 0 <= index < self.state_cache.total_frames:
            frame, meta = self.state_cache.get_frame(index)
            if frame is not None:
                self.video_player.update_frame(frame)
                self.side_panel.update_metrics(meta)
                self.timeline.update_state(
                    index,
                    max(0, self.state_cache.total_frames - 1),
                    False)

    # ════════════════════════════════════════════════════════════════
    #  THREAD 1: READER — reads frames, keeps only the latest
    # ════════════════════════════════════════════════════════════════

    def _reader_loop(self):
        """Read frames on-demand: wait for inference to consume before reading next."""
        while self.is_running:
            # Pause reader if paused OR if we are playing from the cache
            playing_from_cache = self.is_playing and (self.state_cache.current_index < self.state_cache.total_frames - 1)
            if not self.is_playing or playing_from_cache:
                time.sleep(0.02)
                continue

            # Check if inference has consumed the previous frame
            with self._frame_lock:
                slot_full = self._frame_slot is not None

            if slot_full:
                # Inference hasn't consumed yet — wait briefly
                time.sleep(0.003)
                continue

            # Read OUTSIDE the lock to avoid blocking inference
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                self.is_running = False
                self.after(0, self.stop_stream)
                break

            # Atomic slot write
            with self._frame_lock:
                self._raw_frame_index += 1
                self._frames_read += 1
                self._frame_slot = (self._raw_frame_index, frame)
                self._frame_event.set()

    # ════════════════════════════════════════════════════════════════
    #  THREAD 2: INFERENCE — processes frames through the pipeline
    # ════════════════════════════════════════════════════════════════

    def _inference_loop(self):
        """
        Processes every frame through the full cognitive pipeline.
        No frames skipped, no prediction-only shortcuts.
        Every frame gets detection + ReID + tracking + drawing.
        """
        last_inf_time = time.perf_counter()

        while self.is_running:
            self._frame_event.wait(timeout=0.1)
            if not self.is_running:
                break

            # Grab frame (slot is consumed, reader can now read next)
            slot = None
            with self._frame_lock:
                slot = self._frame_slot
                self._frame_slot = None
                self._frame_event.clear()

            if slot is None:
                continue

            frame_idx, frame = slot

            # Full pipeline on EVERY frame — no predict_only shortcuts.
            # predict_only caused box drift, blinking, and stutter.
            t0 = time.perf_counter()
            frame_delta = max(1, frame_idx - self._last_tracker_raw_frame_index)
            annotated, stats = self.pipeline.process_frame(
                frame, frame_delta=frame_delta, frame_index=frame_idx)
            self._last_tracker_raw_frame_index = frame_idx
            t1 = time.perf_counter()
            self._last_display_raw_frame_index = frame_idx

            inf_dt = t1 - last_inf_time
            last_inf_time = t1
            self._inference_fps = 1.0 / (inf_dt + 1e-6)
            self._frames_processed += 1

            stats["inference_fps"] = self._inference_fps
            stats["frames_read"] = self._frames_read
            stats["frames_processed"] = self._frames_processed
            stats["raw_frame_index"] = frame_idx
            stats["frame_delta"] = frame_delta

            if self.evaluator is not None:
                visible_objects = []
                for gid, info in stats.get("track_states", {}).items():
                    state_idx = info.get("identity_state", 0)
                    state_name = "ACTIVE" if state_idx == 0 else "THINKING"
                    visible_objects.append({
                        "global_id": gid,
                        "bbox": info.get("box"),
                        "state": state_name
                    })
                self.evaluator.update(
                    frame_idx=frame_idx, 
                    visible_rendered_identities=visible_objects,
                    tracks=self.pipeline.tracker.tracks,
                    detections=stats.get("raw_detections", []), 
                    suppression_regions=None, # Pipeline doesn't currently expose suppression regions easily
                    frozen_gids=list(self.pipeline.occlusion_manager.frozen_gids)
                )
                
                if frame_idx % 60 == 0:
                    print(f"[EVAL UPDATE] Frame {frame_idx}")
                    rep = self.evaluator.get_final_report()
                    sm = rep["summary"]
                    print("\n[Evaluation]")
                    print(f"Visible Switches: {sm['visible_id_switches']}")
                    print(f"Teleportations: {sm['teleportation_events']}")
                    print(f"Duplicates: {sm['duplicate_box_frames']}")
                    print(f"Stability: {sm['identity_stability_score']:.3f}\n")

            # Cache for timeline scrubbing
            self.state_cache.append(annotated, stats)

            # Publish completed frame for display (atomic swap)
            with self._display_lock:
                self._display_frame = annotated
                self._display_stats = stats

            # Forensic Debug Mode trigger
            if stats.get("id_switches"):
                print(f"[UI] Capturing forensic ID switch event at frame {frame_idx} (Async).")
                # Do forensic capture asynchronously so we don't block inference for too long
                threading.Thread(target=self.forensic.capture_id_switch, 
                                 args=(self.state_cache, stats, annotated), daemon=True).start()

    # ════════════════════════════════════════════════════════════════
    #  MAIN THREAD: DISPLAY — shows completed annotated frames
    # ════════════════════════════════════════════════════════════════

    def _schedule_display(self):
        """Schedule display tick at ~30fps polling rate."""
        if self.is_running:
            self._display_tick()
            self._display_timer = self.after(33, self._schedule_display)
        else:
            self._display_timer = None

    def _display_tick(self):
        """
        Shows the latest pipeline-annotated frame.
        Between inference ticks, re-displays the cached last frame
        so the UI never goes blank.
        """
        if not self.is_running:
            return

        # If playing from cache, ignore new frames and advance cache
        if self.is_playing and self.state_cache.current_index < self.state_cache.total_frames - 1:
            self.state_cache.current_index += 1
            frame, meta = self.state_cache.get_frame(self.state_cache.current_index)
            if frame is not None:
                self.video_player.update_frame(frame)
                self.side_panel.update_metrics(meta)
                self.timeline.update_state(
                    self.state_cache.current_index,
                    max(0, self.state_cache.total_frames - 1),
                    True)
            return

        # Grab latest completed frame (if any)
        new_frame = None
        stats = None
        with self._display_lock:
            if self._display_frame is not None:
                new_frame = self._display_frame
                self._display_frame = None  # Consume
            if self._display_stats is not None:
                stats = self._display_stats
                self._display_stats = None

        # Cache new frame, or keep showing the last one
        if new_frame is not None:
            self._cached_display = new_frame

        if self._cached_display is None:
            return

        self.video_player.update_frame(self._cached_display)

        if stats is not None:
            self.side_panel.update_metrics(stats)

        if self.state_cache.total_frames > 0:
            self.timeline.update_state(
                self.state_cache.current_index,
                max(0, self.state_cache.total_frames - 1),
                self.is_playing)

    # ════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════
    #  BENCHMARK CONTROLS
    # ════════════════════════════════════════════════════════════════

    def _on_benchmark_toggle(self, enabled):
        """Called when user toggles benchmark mode checkbox."""
        if enabled:
            print("[UI] Benchmark mode enabled")
        else:
            print("[UI] Benchmark mode disabled")

    def _on_dataset_change(self, dataset_name, scenario):
        """Called when user changes dataset dropdown."""
        sequences = self.dataset_manager.list_sequences(
            dataset_name, scenario
        )
        self._sequence_map = {s.name: s for s in sequences}
        seq_names = [s.name for s in sequences]
        self.benchmark_panel.update_sequences(seq_names)
        print(f"[UI] Dataset: {dataset_name} | "
              f"Scenario: {scenario} | "
              f"Found {len(sequences)} sequences")

    def _on_benchmark_run(self, dataset, scenario, sequence_name):
        """Run benchmark on selected sequence."""
        # Ensure pipeline is initialized
        if self.pipeline is None:
            self.benchmark_panel.lbl_progress.configure(
                text="Loading pipeline first..."
            )
            self.init_pipeline()

        if self.pipeline is None:
            self.benchmark_panel.lbl_progress.configure(
                text="Pipeline failed to load"
            )
            self.benchmark_panel.btn_run.configure(state="normal")
            self.benchmark_panel.btn_stop.configure(state="disabled")
            return

        # Create benchmark engine if needed
        if self.benchmark_engine is None:
            self.benchmark_engine = BenchmarkEngine(
                self.pipeline, self.dataset_manager, self.forensic
            )

        # Resolve sequence
        seq_info = self._sequence_map.get(sequence_name)
        if seq_info is None:
            # For custom videos, check if it's a direct file pick
            if dataset == "Custom Videos":
                from tkinter import filedialog
                path = filedialog.askopenfilename(
                    title="Select Benchmark Video",
                    filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")],
                )
                if not path:
                    self.benchmark_panel.btn_run.configure(state="normal")
                    self.benchmark_panel.btn_stop.configure(state="disabled")
                    return
                from engine.dataset_manager import SequenceInfo
                seq_info = SequenceInfo(
                    name=os.path.basename(path),
                    dataset="Custom Videos",
                    path=path, frame_dir=None, gt_path=None,
                    frame_rate=30, seq_length=0,
                    im_width=0, im_height=0,
                    is_video=True, video_path=path, tags=["custom"],
                )
            else:
                self.benchmark_panel.lbl_progress.configure(
                    text=f"Sequence '{sequence_name}' not found"
                )
                self.benchmark_panel.btn_run.configure(state="normal")
                self.benchmark_panel.btn_stop.configure(state="disabled")
                return

        # Define callbacks (thread-safe via self.after)
        def progress_cb(fidx, total, stats):
            self.after(0, self.benchmark_panel.update_progress,
                       fidx, total, stats)
            if self.benchmark_engine and self.benchmark_engine.current_result:
                self.after(
                    0, self.benchmark_panel.update_live_results,
                    self.benchmark_engine.current_result
                )

        def frame_cb(annotated, stats):
            self.after(0, self.video_player.update_frame, annotated)
            stats["inference_fps"] = self._inference_fps
            self.after(0, self.side_panel.update_metrics, stats)

        def done_cb(result):
            self.after(0, self.benchmark_panel.show_completed, result)

        # Launch benchmark
        self.benchmark_engine.run_sequence(
            seq_info,
            progress_cb=progress_cb,
            done_cb=done_cb,
            frame_cb=frame_cb
        )

    def _on_benchmark_stop(self):
        """Stop a running benchmark."""
        if self.benchmark_engine:
            self.benchmark_engine.stop()
            print("[UI] Benchmark stopped by user")

    def _on_benchmark_export(self):
        """Export benchmark results."""
        if self.benchmark_engine and self.benchmark_engine.all_results:
            self.benchmark_engine.export_results_csv()
            self.benchmark_engine.export_results_json()
            print("[UI] Benchmark results exported")
            self.benchmark_panel.lbl_progress.configure(
                text="✓ Results exported to logs/"
            )
        else:
            print("[UI] No benchmark results to export")
            self.benchmark_panel.lbl_progress.configure(
                text="No results to export"
            )

    # ════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ════════════════════════════════════════════════════════════════

    def on_closing(self):
        if self.benchmark_engine:
            self.benchmark_engine.stop()
        self.stop_stream()
        if self.pipeline:
            self.pipeline.close()
        self.destroy()


if __name__ == "__main__":
    app = SelfWatchApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
