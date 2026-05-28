import os
import time
import cv2
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk

import config
from ui.video_player import VideoPlayer
from ui.side_panel import SidePanel
from ui.timeline import Timeline
from ui.overlay_panel import OverlayPanel
from ui.overlay_renderer import OverlayRenderer
from engine.async_state_cache import AsyncStateCache
import engine.async_state_cache as _async_cache_mod
from evaluation.evaluator import SELFWATCHEvaluator
from evaluation.experiment_logs.experiment_tracker import ResearchExperimentTracker

# Import multicam
from multicam.multicam_pipeline import MultiCameraPipeline
from multicam.events import CameraEvent, EventType

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SelfWatchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SELFWATCH ΓÇö Multi-Camera Cognitive Tracking Lab")
        self.geometry("1600x900")

        # ΓöÇΓöÇ Core State ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        self.multicam_pipeline = None
        self.state_cache = AsyncStateCache(max_frames=300, jpeg_quality=70)
        self.overlay_renderer = OverlayRenderer()
        
        self.sources = [] # list of sources (int or str)
        
        self.is_running = False
        self.is_playing = True
        self.video_fps = 30.0

        # Thread handles
        self._inference_thread = None
        self._display_timer = None

        # ΓöÇΓöÇ Inference ΓåÆ Display: lock-free latest-state handoff ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        # The inference thread writes here; the display thread reads.
        # No mutex needed ΓÇö Python's GIL makes single-reference
        # assignment atomic. We use overwrite semantics.
        self._latest_frames = None       # latest list of annotated frames
        self._latest_stats = None        # latest list of stats dicts
        self._new_data_ready = False      # flag: new data since last display

        # ΓöÇΓöÇ Counters ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        self._raw_frame_index = 0
        self._inference_fps = 0.0
        self._render_fps = 0.0
        # Phase 3: independent render FPS rolling window
        self._render_fps_window = []
        self._last_render_time = 0.0

        self.video_players = []
        
        self.evaluators = []
        self.experiment_tracker = None
        self.evaluation_enabled = True

        # ΓöÇΓöÇ Diagnostics ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        self._diag_ram_mb = 0.0
        self._diag_q_sizes = ""
        self._diag_drops = 0
        self._diag_inf_ms = 0.0
        self._diag_render_ms = 0.0

        self._build_ui()

    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    #  UI CONSTRUCTION
    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)  # Right panel
        self.grid_rowconfigure(0, weight=1)

        callbacks = {
            "on_source_camera": self.add_camera_source,
            "on_source_video": self.add_video_source,
            "on_start_tracking": self.start_multi_camera,
            "on_toggle_debug": self.toggle_debug,
            "on_play": self.play,
            "on_pause": self.pause,
            "on_next_frame": self.step_forward,
            "on_prev_frame": self.step_backward,
            "on_scrub": self.scrub_to,
            "on_toggle_layer": self.toggle_layer,
            "on_toggle_evaluation": self.toggle_evaluation,
        }

        # Side Panel (Left)
        self.side_panel_scroll = ctk.CTkScrollableFrame(
            self, width=320, fg_color=("gray92", "gray14")
        )
        self.side_panel_scroll.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.side_panel = SidePanel(
            self.side_panel_scroll, callbacks,
            fg_color="transparent"
        )
        self.side_panel.pack(fill="x", expand=False)
        
        # UI Toggle for Evaluation
        self.chk_eval = ctk.CTkCheckBox(
            self.side_panel_scroll, text="Enable Evaluator", 
            command=lambda: self.toggle_evaluation(self.chk_eval.get()))
        self.chk_eval.select()
        self.chk_eval.pack(pady=5, padx=10, anchor="w")

        # Diagnostics Panel
        self.lbl_diag = ctk.CTkLabel(
            self.side_panel_scroll, text="Pipeline Diagnostics",
            font=ctk.CTkFont(weight="bold"))
        self.lbl_diag.pack(pady=(15, 2), padx=10, anchor="w")
        self.lbl_memory = ctk.CTkLabel(
            self.side_panel_scroll,
            text="RAM: -- | Inf: --fps | Rnd: --fps",
            text_color="gray", font=ctk.CTkFont(family="Consolas", size=11))
        self.lbl_memory.pack(pady=2, padx=10, anchor="w")
        self.lbl_queues = ctk.CTkLabel(
            self.side_panel_scroll,
            text="Q: -- | Drops: 0",
            text_color="gray", font=ctk.CTkFont(family="Consolas", size=11))
        self.lbl_queues.pack(pady=2, padx=10, anchor="w")
        self.lbl_encoder = ctk.CTkLabel(
            self.side_panel_scroll,
            text="Enc: --ms | EncQ: 0 | EncDrops: 0",
            text_color="gray", font=ctk.CTkFont(family="Consolas", size=11))
        self.lbl_encoder.pack(pady=2, padx=10, anchor="w")
        
        # Source list display
        self.lbl_sources = ctk.CTkLabel(self.side_panel_scroll, text="Selected Sources:", font=ctk.CTkFont(weight="bold"))
        self.lbl_sources.pack(pady=(10, 2), padx=10, anchor="w")
        self.sources_textbox = ctk.CTkTextbox(self.side_panel_scroll, height=60)
        self.sources_textbox.pack(fill="x", padx=10, pady=2)
        
        self.btn_start = ctk.CTkButton(self.side_panel_scroll, text="Γû╢ Start Multi-Camera", command=self.start_multi_camera, fg_color="#2b7b3b", hover_color="#1e5c2a")
        self.btn_start.pack(pady=10, padx=10, fill="x")

        # Event Log
        self.lbl_events = ctk.CTkLabel(self.side_panel_scroll, text="Cross-Camera Event Log", font=ctk.CTkFont(weight="bold"))
        self.lbl_events.pack(pady=(20, 2), padx=10, anchor="w")
        self.event_logbox = ctk.CTkTextbox(self.side_panel_scroll, height=150, font=ctk.CTkFont(family="Consolas", size=11))
        self.event_logbox.pack(fill="x", padx=10, pady=2)

        # Video Player Container (Center)
        self.video_container = ctk.CTkFrame(self)
        self.video_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=(10, 0))

        # Timeline (Bottom)
        self.timeline = Timeline(self, callbacks, height=80)
        self.timeline.grid(row=1, column=1, sticky="ew", padx=10, pady=10)

        # Overlay Panel (Right)
        self.overlay_panel = OverlayPanel(self, callbacks, width=250)
        self.overlay_panel.grid(row=0, column=2, rowspan=2, sticky="nsew")

    def _log_event(self, event_str):
        self.event_logbox.insert("end", f"{event_str}\n")
        self.event_logbox.see("end")

    def _on_multicam_event(self, event: CameraEvent):
        # Callback from the EventBus
        ev_type = event.event_type.name
        if ev_type == "NEW_GLOBAL":
            msg = f"[NEW] GID{event.global_id} created at CAM{event.camera_id}"
        elif ev_type == "MATCH":
            msg = f"[RECOVER] GID{event.global_id} CAM{event.previous_camera_id} -> CAM{event.camera_id}"
        elif ev_type in ("ENTER", "EXIT"):
            msg = f"[{ev_type}] GID{event.global_id} at CAM{event.camera_id}"
        else:
            msg = f"[{ev_type}] GID{event.global_id}"
            
        self.after(0, self._log_event, msg)

    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    #  SOURCE CONTROL
    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    def add_camera_source(self):
        # Count existing integer sources to auto-increment webcam ID
        cams = [s for s in self.sources if isinstance(s, int)]
        next_cam = len(cams)
        self.sources.append(next_cam)
        self._update_sources_display()

    def add_video_source(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm")],
        )
        if path:
            self.sources.append(path)
            self._update_sources_display()
            
    def _update_sources_display(self):
        self.sources_textbox.delete("1.0", "end")
        for i, src in enumerate(self.sources):
            if isinstance(src, int):
                self.sources_textbox.insert("end", f"CAM{i}: Webcam {src}\n")
            else:
                self.sources_textbox.insert("end", f"CAM{i}: {os.path.basename(src)}\n")

    def start_multi_camera(self):
        if not self.sources:
            self._log_event("Error: No sources selected.")
            return
            
        if self.is_running:
            self.stop_stream()

        self.is_running = True
        
        # Clear container and setup grid
        for widget in self.video_container.winfo_children():
            widget.destroy()
            
        self.video_players = []
        num_cams = len(self.sources)
        cols = min(2, num_cams)
        rows = (num_cams + cols - 1) // cols
        
        for r in range(rows):
            self.video_container.grid_rowconfigure(r, weight=1, uniform="row_group")
        for c in range(cols):
            self.video_container.grid_columnconfigure(c, weight=1, uniform="col_group")
            
        for i in range(num_cams):
            vp = VideoPlayer(self.video_container)
            vp.grid(row=i//cols, column=i%cols, sticky="nsew", padx=2, pady=2)
            vp.show_loading(f"Initializing Camera {i}...")
            self.video_players.append(vp)
            
        threading.Thread(
            target=self._init_stream_worker,
            daemon=True, name="sw-multicam-startup"
        ).start()

    def _init_stream_worker(self):
        t0 = time.perf_counter()
        
        # Initialize MultiCameraPipeline
        selected = self.side_panel.model_var.get()
        variant = "nano"
        if "Medium" in selected: variant = "medium"
        elif "Large" in selected: variant = "large"
        
        self.multicam_pipeline = MultiCameraPipeline(
            detector_variant=variant,
            enable_debug=self.overlay_panel.chk_debug.get()
        )
        
        # Register event listener for UI
        self.multicam_pipeline.event_bus.register_listener(self._on_multicam_event)
        
        # Add cameras
        for i, src in enumerate(self.sources):
            self.multicam_pipeline.add_camera(src, label=f"Source {i}")
            
        # Open cameras
        for cam in self.multicam_pipeline.cameras:
            cam.open()
            
        print(f"[Profiling] Pipeline load time: {time.perf_counter() - t0:.2f}s")
        
        self.after(0, self._finalize_stream_start)

    def _finalize_stream_start(self):
        if not self.is_running:
            return

        self.state_cache.clear()
        
        # Reset counters
        self._raw_frame_index = 0
        self._latest_frames = None
        self._latest_stats = None
        self._new_data_ready = False

        self.evaluators = [SELFWATCHEvaluator() for _ in self.sources]
        self.experiment_tracker = ResearchExperimentTracker()
        
        config_data = {k: v for k, v in vars(config).items() if k.isupper()}
        self.experiment_tracker.save_config(config_data)

        self.is_playing = True
        self.timeline.is_playing = True
        self.timeline.btn_play_pause.configure(text="ΓÅ╕ Pause")

        # Thread B: Inference (decoupled from display)
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="sw-inference")
        self._inference_thread.start()

        # Thread C: Display (runs on main thread via after())
        if self._display_timer is None:
            self._schedule_display()

    def stop_stream(self):
        self.is_running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

        # Stop the async encoder thread cleanly
        self.state_cache.stop()

        if self.multicam_pipeline:
            # We must gracefully close cameras
            for cam in self.multicam_pipeline.cameras:
                cam.close()
            self.multicam_pipeline = None
            
        if self.experiment_tracker:
            for i, ev in enumerate(self.evaluators):
                report = ev.get_final_report()
                # Store report per camera
                self.experiment_tracker.save_metrics(report, suffix=f"_cam{i}")
            avg_fps = self._inference_fps if self._inference_fps else 0.0
            total_time = self._raw_frame_index / avg_fps if avg_fps > 0 else 0.0
            self.experiment_tracker.log_runtime_stats(avg_fps, total_time)

    def toggle_debug(self, state):
        if self.multicam_pipeline:
            for cam in self.multicam_pipeline.cameras:
                cam.pipeline.debug_overlay.enabled = state

    def toggle_layer(self, layer_name, state):
        if self.multicam_pipeline:
            for cam in self.multicam_pipeline.cameras:
                cam.pipeline.layer_manager.toggle(layer_name, state)
            
            # Force redraw if paused
            if not self.is_playing and self._latest_frames:
                self.scrub_to(self.timeline.slider_var.get())

    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    #  PLAYBACK & SCRUBBING
    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    def play(self):
        self.is_playing = True

    def pause(self):
        self.is_playing = False

    def step_forward(self):
        if not self.is_running: return
        self.pause()
        
        if self.state_cache.current_index < self.state_cache.total_frames - 1:
            self.state_cache.current_index += 1
            frames, metas = self.state_cache.get_frame(self.state_cache.current_index)
            self._update_ui_from_cache(frames, metas)
            return

        # Advance one step in multicam pipeline
        results = self.multicam_pipeline.step()
        self._raw_frame_index += 1
        
        frames = [r[0] for r in results]
        stats_list = [r[1] for r in results]
        
        # Combine stats for side panel (use camera 0 or aggregated)
        agg_stats = stats_list[0] if stats_list and stats_list[0] else {}
        
        self.state_cache.append(frames, stats_list)
        self._update_ui_from_cache(frames, stats_list)

    def step_backward(self):
        self.pause()
        if self.state_cache.current_index > 0:
            self.state_cache.current_index -= 1
            frames, metas = self.state_cache.get_frame(self.state_cache.current_index)
            self._update_ui_from_cache(frames, metas)

    def scrub_to(self, index):
        self.pause()
        if 0 <= index < self.state_cache.total_frames:
            frames, metas = self.state_cache.get_frame(index)
            self._update_ui_from_cache(frames, metas)

    def _update_ui_from_cache(self, frames, metas):
        if frames:
            for i, f in enumerate(frames):
                if i < len(self.video_players) and f is not None:
                    meta = metas[i] if (metas and i < len(metas) and metas[i]) else {}
                    render_meta = meta.get("rendering_metadata", {})
                    cam_label = meta.get("camera_label", f"CAM{i}")
                    annotated = self.overlay_renderer.render(f, render_meta, cam_label, i)
                    self.video_players[i].update_frame(annotated)
                    
        # Update metrics using first active camera's stats for now
        # A true shared evaluator will aggregate these later
        for m in metas:
            if m:
                self.side_panel.update_metrics(m)
                break
                
        self.timeline.update_state(
            self.state_cache.current_index,
            max(0, self.state_cache.total_frames - 1),
            False)

    def toggle_evaluation(self, state):
        self.evaluation_enabled = state
        self._log_event(f"Evaluator {'ENABLED' if state else 'DISABLED'}")

    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    #  THREAD B: INFERENCE (decoupled from display)
    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    def _inference_loop(self):
        """
        Inference thread ΓÇö processes frames as fast as GPU allows.

        Architecture:
          - Reads latest frames from camera capture threads (Thread A)
          - Runs detection + tracking + identity sync
          - Writes results to _latest_frames/_latest_stats (atomic)
          - Display thread (Thread C) reads these independently
          - NEVER waits for display thread
          - NEVER blocks on UI
        """
        import psutil
        proc = psutil.Process(os.getpid())

        last_inf_time = time.perf_counter()
        fps_window = []  # Rolling window for smooth FPS
        diag_interval = 20  # Update diagnostics every N frames
        
        while self.is_running:
            # Check pause state ΓÇö if playing from cache, don't run inference
            playing_from_cache = self.is_playing and (
                self.state_cache.current_index < self.state_cache.total_frames - 1)
            if not self.is_playing or playing_from_cache:
                time.sleep(0.02)
                continue

            t_inf_start = time.perf_counter()
            
            # Step the full multi-camera pipeline (non-blocking)
            try:
                results = self.multicam_pipeline.step()
            except Exception as e:
                print(f"[INFERENCE ERROR] {e}")
                time.sleep(0.05)
                continue

            # Check if all streams ended permanently
            if results is None:
                print("[MULTICAM] All streams ended.")
                self.is_running = False
                break

            self._raw_frame_index += 1
                
            frames = [r[0] for r in results]
            stats_list = [r[1] for r in results]
            
            t_inf_end = time.perf_counter()
            inf_dt = t_inf_end - last_inf_time
            last_inf_time = t_inf_end
            
            # Smooth FPS calculation (rolling window)
            fps_window.append(inf_dt)
            if len(fps_window) > 30:
                fps_window.pop(0)
            avg_dt = sum(fps_window) / len(fps_window)
            self._inference_fps = 1.0 / (avg_dt + 1e-6)
            self._diag_inf_ms = (t_inf_end - t_inf_start) * 1000
            
            # Update inference FPS in all stats + run evaluator
            for i, s in enumerate(stats_list):
                if s: 
                    s["inference_fps"] = self._inference_fps
                    
                    if self.evaluation_enabled and i < len(self.evaluators) and self.evaluators[i] is not None:
                        visible_objects = []
                        for pgid, info in s.get("track_states", {}).items():
                            state_idx = info.get("identity_state", 0)
                            state_name = "ACTIVE" if state_idx == 0 else "THINKING"
                            # Remap to global ID
                            mgid = s.get("multicam_active_gids", {}).get(pgid, pgid)
                            visible_objects.append({
                                "global_id": mgid,
                                "bbox": info.get("box"),
                                "state": state_name
                            })
                            
                        cam = self.multicam_pipeline.cameras[i]
                        self.evaluators[i].update(
                            frame_idx=self._raw_frame_index,
                            visible_rendered_identities=visible_objects,
                            tracks=cam.pipeline.tracker.tracks,
                            detections=s.get("raw_detections", []),
                            suppression_regions=None,
                            frozen_gids=list(cam.pipeline.occlusion_manager.frozen_gids),
                            raw_display=s.get("raw_display", None)
                        )
                
            # Append to state cache for scrubbing
            self.state_cache.append(frames, stats_list)

            # ΓöÇΓöÇ Atomic handoff to display thread ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            # No lock needed: Python GIL makes reference assignment atomic.
            # Display thread reads these; if it misses one, it gets the next.
            self._latest_frames = frames
            self._latest_stats = stats_list
            self._new_data_ready = True
                
            # ΓöÇΓöÇ Update diagnostics periodically ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            if self._raw_frame_index % diag_interval == 0:
                try:
                    self._diag_ram_mb = proc.memory_info().rss / 1e6
                    q_sizes = [cam.get_status()["queue_size"]
                               for cam in self.multicam_pipeline.cameras]
                    self._diag_q_sizes = ",".join(map(str, q_sizes))
                    self._diag_drops = sum(
                        cam.dropped_frames
                        for cam in self.multicam_pipeline.cameras)
                except Exception:
                    pass

    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    #  THREAD C: DISPLAY (main thread, paced at 30fps)
    # ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    # Target: 30 FPS render cadence (33ms intervals)
    TARGET_RENDER_FPS = 30
    RENDER_INTERVAL_MS = 1000 // TARGET_RENDER_FPS  # 33ms

    def _schedule_display(self):
        if self.is_running:
            self._display_tick()
            self._display_timer = self.after(
                self.RENDER_INTERVAL_MS, self._schedule_display)
        else:
            self._display_timer = None

    def _display_tick(self):
        """
        Display thread tick ΓÇö runs on the main/UI thread.

        Architecture:
          - Reads latest frames/stats from inference thread (atomic)
          - Renders to VideoPlayer widgets
          - Updates diagnostics panel
          - NEVER blocks inference thread
          - If no new data, displays last known frame (no stall)
        """
        if not self.is_running:
            return

        t_render_start = time.perf_counter()

        # ── Phase 3: Independent render FPS ──────────────────────────
        if self._last_render_time > 0:
            render_dt = t_render_start - self._last_render_time
            self._render_fps_window.append(render_dt)
            if len(self._render_fps_window) > 30:
                self._render_fps_window.pop(0)
            avg_rdt = sum(self._render_fps_window) / len(self._render_fps_window)
            self._render_fps = 1.0 / (avg_rdt + 1e-6)
        self._last_render_time = t_render_start

        # If playing from cache (scrubbing)
        if self.is_playing and self.state_cache.current_index < self.state_cache.total_frames - 1:
            self.state_cache.current_index += 1
            frames, metas = self.state_cache.get_frame(self.state_cache.current_index)
            self._update_ui_from_cache(frames, metas)
            self.timeline.is_playing = True
            return

        # ΓöÇΓöÇ Read latest data (non-blocking, no lock) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        if self._new_data_ready:
            new_frames = self._latest_frames
            new_stats = self._latest_stats
            self._new_data_ready = False
        else:
            new_frames = None
            new_stats = None

        # ΓöÇΓöÇ Render frames to video players ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        if new_frames is not None:
            for i, f in enumerate(new_frames):
                if i < len(self.video_players) and f is not None:
                    meta = new_stats[i] if (new_stats and i < len(new_stats) and new_stats[i]) else {}
                    render_meta = meta.get("rendering_metadata", {})
                    cam_label = meta.get("camera_label", f"CAM{i}")
                    annotated = self.overlay_renderer.render(f, render_meta, cam_label, i)
                    self.video_players[i].update_frame(annotated)

        # ΓöÇΓöÇ Update side panel metrics ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        if new_stats is not None:
            try:
                reg_stats = self.multicam_pipeline.global_registry.get_stats()
                display_stat = None
                for s in new_stats:
                    if s:
                        display_stat = s
                        break
                
                if display_stat:
                    display_stat["active_tracks"] = reg_stats["active_global_ids"]
                    self.side_panel.update_metrics(display_stat)
            except Exception:
                pass

        # ΓöÇΓöÇ Update timeline ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        if self.state_cache.total_frames > 0:
            self.timeline.update_state(
                self.state_cache.current_index,
                max(0, self.state_cache.total_frames - 1),
                self.is_playing)

        # ΓöÇΓöÇ Update diagnostics (cheap, every tick) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        t_render_end = time.perf_counter()
        self._diag_render_ms = (t_render_end - t_render_start) * 1000

        try:
            self.lbl_memory.configure(
                text=f"RAM: {self._diag_ram_mb:.0f}MB | "
                     f"Inf: {self._inference_fps:.0f}fps "
                     f"({self._diag_inf_ms:.0f}ms) | "
                     f"Rnd: {self._render_fps:.0f}fps")
            self.lbl_queues.configure(
                text=f"Q: [{self._diag_q_sizes}] | "
                     f"Drops: {self._diag_drops} | "
                     f"RndMs: {self._diag_render_ms:.1f}ms")
            # Phase 1+3: Async encoder diagnostics (module-level import, no overhead)
            self.lbl_encoder.configure(
                text=f"Enc: {_async_cache_mod.AsyncStateCache.encode_ms_avg:.1f}ms | "
                     f"EncQ: {_async_cache_mod.AsyncStateCache.encode_queue_size} | "
                     f"EncDrops: {_async_cache_mod.AsyncStateCache.encode_drops}")
        except Exception:
            pass

    def on_closing(self):
        self.stop_stream()
        self.destroy()

if __name__ == "__main__":
    app = SelfWatchApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
