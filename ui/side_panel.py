import customtkinter as ctk

class SidePanel(ctk.CTkFrame):
    def __init__(self, master, callbacks, **kwargs):
        super().__init__(master, **kwargs)
        self.callbacks = callbacks
        
        # ── Model Selector ──
        lbl_model = ctk.CTkLabel(self, text="Model", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_model.pack(pady=(15, 5), padx=10, anchor="w")

        self.model_var = ctk.StringVar(value="RT-DETR-Nano")
        self.model_dropdown = ctk.CTkOptionMenu(
            self, variable=self.model_var,
            values=["RT-DETR-Nano", "RT-DETR-Medium", "RT-DETR-Large"],
        )
        self.model_dropdown.pack(pady=5, padx=10, fill="x")

        # ── Data Sources ──
        lbl_source = ctk.CTkLabel(self, text="Input Source", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_source.pack(pady=(15, 5), padx=10, anchor="w")
        
        self.btn_camera = ctk.CTkButton(self, text="🎥  Use Camera", command=self._on_camera)
        self.btn_camera.pack(pady=5, padx=10, fill="x")
        
        self.btn_video = ctk.CTkButton(self, text="📂  Analyze Video...", command=self._on_video)
        self.btn_video.pack(pady=5, padx=10, fill="x")
        
        # ── Performance ──
        lbl_perf = ctk.CTkLabel(self, text="Performance", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_perf.pack(pady=(25, 5), padx=10, anchor="w")
        
        self.lbl_inference_fps = ctk.CTkLabel(self, text="Inference FPS: --", anchor="w",
                                               text_color="#4CAF50", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_inference_fps.pack(pady=2, padx=15, fill="x")
        
        self.lbl_profiling = ctk.CTkLabel(self, text="Det: -- | ReID: --", anchor="w", text_color="gray")
        self.lbl_profiling.pack(pady=2, padx=15, fill="x")
        
        self.lbl_frames_info = ctk.CTkLabel(self, text="Read: 0 | Processed: 0", anchor="w", text_color="gray")
        self.lbl_frames_info.pack(pady=2, padx=15, fill="x")

        # ── Live Metrics ──
        self.metrics_expanded = True
        self.btn_metrics_toggle = ctk.CTkButton(
            self, text="▼ Cognitive Metrics", 
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="transparent", text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"), anchor="w",
            command=self._toggle_metrics
        )
        self.btn_metrics_toggle.pack(pady=(20, 0), padx=5, fill="x")
        
        self.metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_frame.pack(fill="x", padx=0, pady=0)
        
        self.lbl_tracks = ctk.CTkLabel(self.metrics_frame, text="Active Tracks: 0", anchor="w")
        self.lbl_tracks.pack(pady=2, padx=15, fill="x")
        
        self.lbl_phantoms = ctk.CTkLabel(self.metrics_frame, text="Phantoms: 0", anchor="w")
        self.lbl_phantoms.pack(pady=2, padx=15, fill="x")
        
        self.lbl_resurrections = ctk.CTkLabel(self.metrics_frame, text="Resurrections: 0", anchor="w")
        self.lbl_resurrections.pack(pady=2, padx=15, fill="x")

        self.lbl_warm_mem = ctk.CTkLabel(self.metrics_frame, text="Warm Memory: 0", anchor="w")
        self.lbl_warm_mem.pack(pady=2, padx=15, fill="x")
        
        self.lbl_locks = ctk.CTkLabel(self.metrics_frame, text="Hard Locks: 0 | Soft: 0", anchor="w")
        self.lbl_locks.pack(pady=2, padx=15, fill="x")
        
        # Removed toggles to OverlayPanel
        
    def _toggle_metrics(self):
        self.metrics_expanded = not self.metrics_expanded
        if self.metrics_expanded:
            self.btn_metrics_toggle.configure(text="▼ Cognitive Metrics")
            self.metrics_frame.pack(fill="x", padx=0, pady=0, after=self.btn_metrics_toggle)
        else:
            self.btn_metrics_toggle.configure(text="▶ Cognitive Metrics")
            self.metrics_frame.pack_forget()
        
    def _on_camera(self):
        if self.callbacks.get("on_source_camera"):
            self.callbacks["on_source_camera"]()
            
    def _on_video(self):
        if self.callbacks.get("on_source_video"):
            self.callbacks["on_source_video"]()

    def _on_layer(self, layer_name, chk_box):
        if self.callbacks.get("on_toggle_layer"):
            self.callbacks["on_toggle_layer"](layer_name, bool(chk_box.get()))
            
    def update_metrics(self, stats):
        """Update labels from engine stats."""
        if not stats:
            return

        # Performance
        inf_fps = stats.get("inference_fps", 0)
        self.lbl_inference_fps.configure(text=f"Inference FPS: {inf_fps:.1f}")
        
        self.lbl_profiling.configure(
            text=f"Det: {stats.get('det_ms', 0):.0f}ms | ReID: {stats.get('reid_ms', 0):.0f}ms | Trk: {stats.get('trk_ms', 0):.0f}ms"
        )
        
        fr = stats.get("frames_read", 0)
        fp = stats.get("frames_processed", 0)
        skipped = fr - fp
        self.lbl_frames_info.configure(text=f"Read: {fr} | Processed: {fp} | Skipped: {skipped}")

        # Cognitive
        self.lbl_tracks.configure(text=f"Active Tracks: {stats.get('active_tracks', 0)}")
        self.lbl_phantoms.configure(text=f"Phantoms: {stats.get('phantom_count', 0)}")
        
        if "metrics" in stats:
            m = stats["metrics"]
            self.lbl_resurrections.configure(text=f"Resurrections: {m.retrieval_successes}")
            self.lbl_locks.configure(text=f"Hard Locks: {m.hard_locks} | Soft: {m.soft_locks}")
            self.lbl_warm_mem.configure(text=f"Warm Memory: {m.memory_saves}")
