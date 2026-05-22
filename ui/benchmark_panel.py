"""
SELFWATCH — Benchmark Panel UI

Research-grade benchmarking controls and live results display.
Integrated into the left side panel as a collapsible section with:
  - Dataset/Scenario selectors
  - Benchmark Mode toggle
  - Run/Stop/Export controls
  - Live metrics display with progress
  - Failure taxonomy breakdown
  - Result history
"""

import os
import customtkinter as ctk


class BenchmarkPanel(ctk.CTkFrame):
    """
    Collapsible benchmark control panel for the SELFWATCH side panel.
    
    Provides dataset selection, scenario filtering, benchmark execution
    controls, and a live results display.
    """

    def __init__(self, master, callbacks=None, **kwargs):
        super().__init__(master, **kwargs)
        self.callbacks = callbacks or {}
        self.is_expanded = True

        # ── Header (always visible) ──────────────────────────────────
        self.btn_toggle = ctk.CTkButton(
            self, text="▼ Benchmark Mode",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="transparent", text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"), anchor="w",
            command=self._toggle_expand
        )
        self.btn_toggle.pack(pady=(15, 0), padx=5, fill="x")

        # ── Collapsible Content ──────────────────────────────────────
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="x", padx=0, pady=0)

        sub_font = ctk.CTkFont(size=12)
        small_font = ctk.CTkFont(size=11)

        # Benchmark Mode Toggle
        self.benchmark_enabled = ctk.BooleanVar(value=False)
        self.chk_benchmark = ctk.CTkCheckBox(
            self.content, text="Enable Benchmark Mode",
            variable=self.benchmark_enabled,
            command=self._on_toggle_benchmark,
            font=sub_font
        )
        self.chk_benchmark.pack(pady=(8, 4), padx=10, anchor="w")

        # ── Dataset Selector ─────────────────────────────────────────
        lbl_ds = ctk.CTkLabel(
            self.content, text="Dataset",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        lbl_ds.pack(pady=(8, 2), padx=10, anchor="w")

        self.dataset_var = ctk.StringVar(value="Custom Videos")
        self.dataset_dropdown = ctk.CTkOptionMenu(
            self.content, variable=self.dataset_var,
            values=["DanceTrack", "MOT20", "Custom Videos"],
            command=self._on_dataset_change,
            font=small_font, width=200
        )
        self.dataset_dropdown.pack(pady=2, padx=10, fill="x")

        # ── Scenario Selector ────────────────────────────────────────
        lbl_sc = ctk.CTkLabel(
            self.content, text="Scenario Filter",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        lbl_sc.pack(pady=(8, 2), padx=10, anchor="w")

        self.scenario_var = ctk.StringVar(value="all")
        self.scenario_dropdown = ctk.CTkOptionMenu(
            self.content, variable=self.scenario_var,
            values=["all", "crowd", "crossing", "occlusion", "re-entry"],
            font=small_font, width=200
        )
        self.scenario_dropdown.pack(pady=2, padx=10, fill="x")

        # ── Sequence Selector ────────────────────────────────────────
        lbl_seq = ctk.CTkLabel(
            self.content, text="Sequence",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        lbl_seq.pack(pady=(8, 2), padx=10, anchor="w")

        self.sequence_var = ctk.StringVar(value="(select dataset first)")
        self.sequence_dropdown = ctk.CTkOptionMenu(
            self.content, variable=self.sequence_var,
            values=["(select dataset first)"],
            font=small_font, width=200
        )
        self.sequence_dropdown.pack(pady=2, padx=10, fill="x")

        # ── Run / Stop Buttons ───────────────────────────────────────
        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.pack(pady=(10, 2), padx=10, fill="x")

        self.btn_run = ctk.CTkButton(
            btn_frame, text="▶  Run Benchmark",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2D7D46", hover_color="#236B38",
            command=self._on_run, width=120
        )
        self.btn_run.pack(side="left", padx=(0, 5), expand=True, fill="x")

        self.btn_stop = ctk.CTkButton(
            btn_frame, text="■  Stop",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#8B2020", hover_color="#6B1515",
            command=self._on_stop, width=80, state="disabled"
        )
        self.btn_stop.pack(side="right", padx=(5, 0))

        # ── Progress ─────────────────────────────────────────────────
        self.progress_bar = ctk.CTkProgressBar(
            self.content, width=200, height=12
        )
        self.progress_bar.pack(pady=(8, 2), padx=10, fill="x")
        self.progress_bar.set(0)

        self.lbl_progress = ctk.CTkLabel(
            self.content, text="Ready", anchor="w",
            font=small_font, text_color="gray"
        )
        self.lbl_progress.pack(pady=1, padx=12, anchor="w")

        # ── Live Results Display ─────────────────────────────────────
        self.results_frame = ctk.CTkFrame(
            self.content, fg_color=("gray85", "gray17"),
            corner_radius=8
        )
        self.results_frame.pack(pady=(8, 2), padx=8, fill="x")

        # Result labels
        res_font = ctk.CTkFont(size=11)
        self.lbl_res_fps = ctk.CTkLabel(
            self.results_frame, text="Avg FPS: --",
            font=res_font, anchor="w"
        )
        self.lbl_res_fps.pack(pady=1, padx=8, fill="x")

        self.lbl_res_switches = ctk.CTkLabel(
            self.results_frame, text="ID Switches: --",
            font=res_font, anchor="w", text_color="#FF6B6B"
        )
        self.lbl_res_switches.pack(pady=1, padx=8, fill="x")

        self.lbl_res_frags = ctk.CTkLabel(
            self.results_frame, text="Fragmentations: --",
            font=res_font, anchor="w"
        )
        self.lbl_res_frags.pack(pady=1, padx=8, fill="x")

        self.lbl_res_recoveries = ctk.CTkLabel(
            self.results_frame, text="Recoveries: --",
            font=res_font, anchor="w", text_color="#4CAF50"
        )
        self.lbl_res_recoveries.pack(pady=1, padx=8, fill="x")

        self.lbl_res_cbiou = ctk.CTkLabel(
            self.results_frame, text="C-BIoU Saves: --",
            font=res_font, anchor="w"
        )
        self.lbl_res_cbiou.pack(pady=1, padx=8, fill="x")

        self.lbl_res_suppress = ctk.CTkLabel(
            self.results_frame, text="Suppressions: --",
            font=res_font, anchor="w"
        )
        self.lbl_res_suppress.pack(pady=1, padx=8, fill="x")

        self.lbl_res_continuity = ctk.CTkLabel(
            self.results_frame, text="Continuity: --",
            font=res_font, anchor="w", text_color="#64B5F6"
        )
        self.lbl_res_continuity.pack(pady=1, padx=8, fill="x")

        self.lbl_res_lifetime = ctk.CTkLabel(
            self.results_frame, text="Avg Lifetime: --",
            font=res_font, anchor="w"
        )
        self.lbl_res_lifetime.pack(pady=(1, 4), padx=8, fill="x")

        # ── Failure Taxonomy ─────────────────────────────────────────
        lbl_tax = ctk.CTkLabel(
            self.content, text="Failure Taxonomy",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        lbl_tax.pack(pady=(10, 2), padx=10, anchor="w")

        self.taxonomy_frame = ctk.CTkFrame(
            self.content, fg_color=("gray85", "gray17"),
            corner_radius=8
        )
        self.taxonomy_frame.pack(pady=2, padx=8, fill="x")

        tax_font = ctk.CTkFont(size=11)
        self.lbl_tax_cross = ctk.CTkLabel(
            self.taxonomy_frame, text="  Crossing: 0",
            font=tax_font, anchor="w", text_color="#FF9800"
        )
        self.lbl_tax_cross.pack(pady=1, padx=8, fill="x")

        self.lbl_tax_occl = ctk.CTkLabel(
            self.taxonomy_frame, text="  Occlusion: 0",
            font=tax_font, anchor="w", text_color="#FF9800"
        )
        self.lbl_tax_occl.pack(pady=1, padx=8, fill="x")

        self.lbl_tax_frag = ctk.CTkLabel(
            self.taxonomy_frame, text="  Fragmentation: 0",
            font=tax_font, anchor="w", text_color="#FF9800"
        )
        self.lbl_tax_frag.pack(pady=1, padx=8, fill="x")

        self.lbl_tax_amb = ctk.CTkLabel(
            self.taxonomy_frame, text="  Ambiguity: 0",
            font=tax_font, anchor="w", text_color="#FF9800"
        )
        self.lbl_tax_amb.pack(pady=1, padx=8, fill="x")

        self.lbl_tax_supp = ctk.CTkLabel(
            self.taxonomy_frame, text="  Suppression Expiry: 0",
            font=tax_font, anchor="w", text_color="#FF9800"
        )
        self.lbl_tax_supp.pack(pady=(1, 4), padx=8, fill="x")

        # ── Export Button ────────────────────────────────────────────
        self.btn_export = ctk.CTkButton(
            self.content, text="📊  Export Results",
            font=ctk.CTkFont(size=12),
            fg_color="#1565C0", hover_color="#0D47A1",
            command=self._on_export
        )
        self.btn_export.pack(pady=(10, 5), padx=10, fill="x")

        # Start collapsed
        self._toggle_expand()

    # ── Toggle ───────────────────────────────────────────────────────

    def _toggle_expand(self):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.btn_toggle.configure(text="▼ Benchmark Mode")
            self.content.pack(fill="x", padx=0, pady=0,
                            after=self.btn_toggle)
        else:
            self.btn_toggle.configure(text="▶ Benchmark Mode")
            self.content.pack_forget()

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_toggle_benchmark(self):
        enabled = self.benchmark_enabled.get()
        cb = self.callbacks.get("on_benchmark_toggle")
        if cb:
            cb(enabled)

    def _on_dataset_change(self, value):
        cb = self.callbacks.get("on_dataset_change")
        if cb:
            cb(value, self.scenario_var.get())

    def _on_run(self):
        cb = self.callbacks.get("on_benchmark_run")
        if cb:
            self.btn_run.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.progress_bar.set(0)
            self.lbl_progress.configure(text="Starting benchmark...")
            cb(
                self.dataset_var.get(),
                self.scenario_var.get(),
                self.sequence_var.get()
            )

    def _on_stop(self):
        cb = self.callbacks.get("on_benchmark_stop")
        if cb:
            cb()
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_progress.configure(text="Stopped")

    def _on_export(self):
        cb = self.callbacks.get("on_benchmark_export")
        if cb:
            cb()

    # ── Update Methods (called from app.py) ──────────────────────────

    def update_sequences(self, sequence_names):
        """Update the sequence dropdown with available sequences."""
        if not sequence_names:
            sequence_names = ["(no sequences found)"]
        self.sequence_dropdown.configure(values=sequence_names)
        self.sequence_var.set(sequence_names[0])

    def update_progress(self, frame_idx, total_frames, stats):
        """Update progress bar and live stats during benchmark."""
        if total_frames > 0:
            pct = min(frame_idx / total_frames, 1.0)
            self.progress_bar.set(pct)
            self.lbl_progress.configure(
                text=f"Frame {frame_idx}/{total_frames} "
                     f"({pct:.0%})"
            )

    def update_live_results(self, result):
        """Update results panel from a BenchmarkResult object."""
        self.lbl_res_fps.configure(text=f"Avg FPS: {result.avg_fps:.1f}")
        self.lbl_res_switches.configure(
            text=f"ID Switches: {result.id_switches}"
        )
        self.lbl_res_frags.configure(
            text=f"Fragmentations: {result.fragmentations}"
        )
        self.lbl_res_recoveries.configure(
            text=f"Recoveries: {result.recoveries}"
        )
        self.lbl_res_cbiou.configure(
            text=f"C-BIoU Saves: {result.cbiou_recoveries}"
        )
        self.lbl_res_suppress.configure(
            text=f"Suppressions: {result.suppression_events}"
        )
        cont = result.identity_continuity
        self.lbl_res_continuity.configure(
            text=f"Continuity: {cont:.2%}" if cont else "Continuity: --"
        )
        self.lbl_res_lifetime.configure(
            text=f"Avg Lifetime: {result.avg_track_lifetime:.0f}f"
        )

        # Taxonomy
        fc = result.failure_counts
        self.lbl_tax_cross.configure(
            text=f"  Crossing: {fc.get('crossing', 0)}"
        )
        self.lbl_tax_occl.configure(
            text=f"  Occlusion: {fc.get('occlusion', 0)}"
        )
        self.lbl_tax_frag.configure(
            text=f"  Fragmentation: {fc.get('fragmentation', 0)}"
        )
        self.lbl_tax_amb.configure(
            text=f"  Ambiguity: {fc.get('ambiguity', 0)}"
        )
        self.lbl_tax_supp.configure(
            text=f"  Suppression Expiry: {fc.get('suppression_expiry', 0)}"
        )

    def show_completed(self, result):
        """Called when benchmark finishes."""
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.progress_bar.set(1.0)
        self.lbl_progress.configure(
            text=f"✓ Complete — {result.total_frames}f in "
                 f"{result.runtime_seconds:.1f}s"
        )
        self.update_live_results(result)
