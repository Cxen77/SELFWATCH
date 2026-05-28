import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2


class VideoPlayer(ctk.CTkFrame):
    """
    Phase 3: Optimized video display widget.

    Optimizations applied:
      - Caches target (w, h) dimensions via <Configure> event — no winfo_width/height every frame
      - Skips BGR→RGB + resize + ImageTk conversion when the frame pointer (id) is unchanged
      - Channel flip via numpy slice [::-1] instead of cv2.cvtColor (faster for display-only path)
      - INTER_LINEAR resize (fastest acceptable quality for display)
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Frame for the video canvas
        self.canvas_frame = ctk.CTkFrame(self, fg_color="black")
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Use standard tk.Label for high-FPS video streaming
        # (CTkLabel has GC bugs with rapid CTkImage updates)
        self.video_label = tk.Label(
            self.canvas_frame, text="No Video Loaded",
            fg="gray", bg="black", font=("Arial", 14))
        self.video_label.pack(fill="both", expand=True)

        self.current_image = None

        # ── Phase 3: Render caching ──────────────────────────────────────
        # Track last rendered frame by object id to skip stale renders
        self._last_frame_id: int = -1
        # Cached display dimensions — updated only on <Configure> events
        self._cached_display_w: int = 0
        self._cached_display_h: int = 0

        # Bind resize event to refresh dimension cache (not per-frame)
        self.canvas_frame.bind("<Configure>", self._on_resize)

    # ─── Callbacks ───────────────────────────────────────────────────────

    def _on_resize(self, event):
        """Update cached display dimensions and invalidate last frame on resize."""
        self._cached_display_w = event.width
        self._cached_display_h = event.height
        # Force a re-render on next frame since display size changed
        self._last_frame_id = -1

    def show_loading(self, text="Loading..."):
        self.current_image = None
        self._last_frame_id = -1
        self.video_label.configure(image="", text=text)

    # ─── Main render path ─────────────────────────────────────────────────

    def update_frame(self, cv2_image):
        """
        Display a cv2 BGR frame in the widget.

        Phase 3 optimization:
          - If the same numpy array is submitted again (same id()), returns immediately.
          - Uses cached (w, h) from <Configure> events instead of winfo calls per frame.
          - Channel flip via [::-1] slice instead of cv2.cvtColor.
          - INTER_LINEAR resize for best speed/quality tradeoff at display resolution.
        """
        if cv2_image is None:
            return

        # ── Stale-frame skip ────────────────────────────────────────────
        # id() changes every time inference produces a new numpy array.
        # Same id → same object → display thread already showed this frame.
        frame_id = id(cv2_image)
        if frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id

        # ── Display dimensions (cached) ─────────────────────────────────
        target_w = self._cached_display_w
        target_h = self._cached_display_h
        if target_w <= 1 or target_h <= 1:
            # First call before <Configure> fires — fall back to winfo (once only)
            target_w = self.canvas_frame.winfo_width()
            target_h = self.canvas_frame.winfo_height()
        if target_w <= 1 or target_h <= 1:
            target_w, target_h = 800, 600

        # ── BGR → RGB via numpy channel flip (no copy if array is contiguous) ──
        rgb_image = cv2_image[:, :, ::-1]

        # ── Scale to fit while preserving aspect ratio ──────────────────
        img_h, img_w = rgb_image.shape[:2]
        ratio = min(target_w / img_w, target_h / img_h)
        new_w, new_h = max(1, int(img_w * ratio)), max(1, int(img_h * ratio))

        rgb_image = cv2.resize(rgb_image, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)

        # ── Tkinter display ─────────────────────────────────────────────
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)

        self.video_label.configure(image=tk_image, text="")
        # Hold reference to prevent garbage collection
        self.current_image = tk_image
