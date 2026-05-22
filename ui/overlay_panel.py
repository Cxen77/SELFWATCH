import customtkinter as ctk

class OverlayPanel(ctk.CTkFrame):
    def __init__(self, master, callbacks, **kwargs):
        super().__init__(master, **kwargs)
        self.callbacks = callbacks

        # ── Toggles ──
        self.lbl_toggles = ctk.CTkLabel(self, text="Overlays", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_toggles.pack(pady=(25, 5), padx=10, anchor="w")
        
        # Scrollable Frame for Overlays
        self.overlays_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        self.overlays_frame.pack(fill="both", expand=True, padx=5, pady=0)

        self.chk_debug = ctk.CTkCheckBox(self.overlays_frame, text="Memory Debug Info", command=self._on_toggle_debug)
        self.chk_debug.pack(pady=2, padx=10, anchor="w")
        self.chk_debug.select()

        sub_font = ctk.CTkFont(size=11)

        # Tracking Layer
        self.chk_trk = ctk.CTkCheckBox(self.overlays_frame, text="Tracking Layer", command=lambda: self._on_layer("tracking", self.chk_trk))
        self.chk_trk.pack(pady=(5, 0), padx=10, anchor="w")
        self.chk_trk.select()
        self.chk_trk_id = ctk.CTkCheckBox(self.overlays_frame, text="  IDs", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("tracking_ids", self.chk_trk_id))
        self.chk_trk_id.pack(pady=1, padx=25, anchor="w")
        self.chk_trk_id.select()
        self.chk_trk_age = ctk.CTkCheckBox(self.overlays_frame, text="  Age", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("tracking_age", self.chk_trk_age))
        self.chk_trk_age.pack(pady=1, padx=25, anchor="w")
        self.chk_trk_age.select()

        # Motion Layer
        self.chk_mot = ctk.CTkCheckBox(self.overlays_frame, text="Motion Layer", command=lambda: self._on_layer("motion", self.chk_mot))
        self.chk_mot.pack(pady=(5, 0), padx=10, anchor="w")
        
        self.chk_mot_vel = ctk.CTkCheckBox(self.overlays_frame, text="  Velocity", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("motion_velocity", self.chk_mot_vel))
        self.chk_mot_vel.pack(pady=1, padx=25, anchor="w")
        
        self.chk_mot_pred = ctk.CTkCheckBox(self.overlays_frame, text="  Prediction", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("motion_prediction", self.chk_mot_pred))
        self.chk_mot_pred.pack(pady=1, padx=25, anchor="w")
        

        # Association Layer
        self.chk_asc = ctk.CTkCheckBox(self.overlays_frame, text="Association Layer", command=lambda: self._on_layer("association", self.chk_asc))
        self.chk_asc.pack(pady=(5, 0), padx=10, anchor="w")
        
        self.chk_asc_cost = ctk.CTkCheckBox(self.overlays_frame, text="  Cost Labels", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("assoc_cost", self.chk_asc_cost))
        self.chk_asc_cost.pack(pady=1, padx=25, anchor="w")
        
        self.chk_asc_cbiou = ctk.CTkCheckBox(self.overlays_frame, text="  C-BIoU Labels", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("assoc_cbiou", self.chk_asc_cbiou))
        self.chk_asc_cbiou.pack(pady=1, padx=25, anchor="w")
        
        self.chk_asc_meth = ctk.CTkCheckBox(self.overlays_frame, text="  Match Type", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("assoc_method", self.chk_asc_meth))
        self.chk_asc_meth.pack(pady=1, padx=25, anchor="w")
        
        self.chk_asc_amb = ctk.CTkCheckBox(self.overlays_frame, text="  Ambiguity", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("assoc_ambiguity", self.chk_asc_amb))
        self.chk_asc_amb.pack(pady=1, padx=25, anchor="w")
        

        # Cognitive Layer
        self.chk_cog = ctk.CTkCheckBox(self.overlays_frame, text="Cognitive Layer", command=lambda: self._on_layer("cognitive", self.chk_cog))
        self.chk_cog.pack(pady=(5, 0), padx=10, anchor="w")
        self.chk_cog.select()
        self.chk_cog_froz = ctk.CTkCheckBox(self.overlays_frame, text="  Frozen", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("cog_frozen", self.chk_cog_froz))
        self.chk_cog_froz.pack(pady=1, padx=25, anchor="w")
        self.chk_cog_froz.select()
        self.chk_cog_thnk = ctk.CTkCheckBox(self.overlays_frame, text="  Thinking", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("cog_thinking", self.chk_cog_thnk))
        self.chk_cog_thnk.pack(pady=1, padx=25, anchor="w")
        self.chk_cog_thnk.select()

        # Forensic Layer
        self.chk_for = ctk.CTkCheckBox(self.overlays_frame, text="Forensic Layer", command=lambda: self._on_layer("forensic", self.chk_for))
        self.chk_for.pack(pady=(5, 0), padx=10, anchor="w")
        self.chk_for.select()
        self.chk_for_supp = ctk.CTkCheckBox(self.overlays_frame, text="  Suppression Zones", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("for_suppress", self.chk_for_supp))
        self.chk_for_supp.pack(pady=1, padx=25, anchor="w")
        
        self.chk_for_fail = ctk.CTkCheckBox(self.overlays_frame, text="  Failure Events", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("for_failure", self.chk_for_fail))
        self.chk_for_fail.pack(pady=1, padx=25, anchor="w")
        self.chk_for_fail.select()
        
        self.chk_for_own = ctk.CTkCheckBox(self.overlays_frame, text="  Ownership Transfers", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("for_ownership_transfers", self.chk_for_own))
        self.chk_for_own.pack(pady=1, padx=25, anchor="w")

        self.chk_for_frag = ctk.CTkCheckBox(self.overlays_frame, text="  Fragmentation Warnings", checkbox_width=18, checkbox_height=18, font=sub_font, command=lambda: self._on_layer("for_fragmentation", self.chk_for_frag))
        self.chk_for_frag.pack(pady=1, padx=25, anchor="w")
        
    def toggle_master_checkbox(self, layer_name):
        mapping = {
            "tracking": self.chk_trk,
            "motion": self.chk_mot,
            "association": self.chk_asc,
            "cognitive": self.chk_cog,
            "forensic": self.chk_for
        }
        if layer_name in mapping:
            chk = mapping[layer_name]
            chk.toggle()
            self._on_layer(layer_name, chk)

    def _on_toggle_debug(self):
        if self.callbacks.get("on_toggle_debug"):
            self.callbacks["on_toggle_debug"](bool(self.chk_debug.get()))

    def _on_layer(self, layer_name, chk_box):
        if self.callbacks.get("on_toggle_layer"):
            self.callbacks["on_toggle_layer"](layer_name, bool(chk_box.get()))
