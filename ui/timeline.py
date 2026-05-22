import customtkinter as ctk

class Timeline(ctk.CTkFrame):
    def __init__(self, master, callbacks, **kwargs):
        super().__init__(master, **kwargs)
        self.callbacks = callbacks
        
        # Play/Pause Button
        self.btn_play_pause = ctk.CTkButton(
            self, text="⏸ Pause", width=80, 
            command=self._toggle_play
        )
        self.btn_play_pause.pack(side="left", padx=5, pady=5)
        
        # Previous Frame
        self.btn_prev = ctk.CTkButton(
            self, text="⏪ Prev", width=60, 
            command=self._on_prev
        )
        self.btn_prev.pack(side="left", padx=5, pady=5)
        
        # Next Frame
        self.btn_next = ctk.CTkButton(
            self, text="Next ⏩", width=60, 
            command=self._on_next
        )
        self.btn_next.pack(side="left", padx=5, pady=5)
        
        # Scrub Bar
        self.slider = ctk.CTkSlider(self, from_=0, to=100, command=self._on_slider)
        self.slider.set(0)
        self.slider.pack(side="left", fill="x", expand=True, padx=15, pady=5)
        
        # Frame Counter
        self.lbl_frame = ctk.CTkLabel(self, text="Frame: 0 / 0", width=120)
        self.lbl_frame.pack(side="left", padx=10)
        
        self.is_playing = True
        
    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play_pause.configure(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing and self.callbacks.get("on_play"):
            self.callbacks["on_play"]()
        elif not self.is_playing and self.callbacks.get("on_pause"):
            self.callbacks["on_pause"]()
            
    def _on_prev(self):
        if self.is_playing:
            self._toggle_play()
        if self.callbacks.get("on_prev_frame"):
            self.callbacks["on_prev_frame"]()
            
    def _on_next(self):
        if self.is_playing:
            self._toggle_play()
        if self.callbacks.get("on_next_frame"):
            self.callbacks["on_next_frame"]()
            
    def _on_slider(self, value):
        if self.is_playing:
            self._toggle_play()
        if self.callbacks.get("on_scrub"):
            self.callbacks["on_scrub"](int(value))
            
    def update_state(self, current_frame, total_frames, is_playing):
        """Update slider and labels from the engine."""
        self.lbl_frame.configure(text=f"Frame: {current_frame} / {total_frames}")
        self.slider.configure(to=max(1, total_frames))
        self.slider.set(current_frame)
        
        if self.is_playing != is_playing:
            self.is_playing = is_playing
            self.btn_play_pause.configure(text="⏸ Pause" if self.is_playing else "▶ Play")
