import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

class VideoPlayer(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Frame for the video canvas
        self.canvas_frame = ctk.CTkFrame(self, fg_color="black")
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Use standard tk.Label for high-FPS video streaming (CTkLabel has GC bugs with rapid CTkImage updates)
        self.video_label = tk.Label(self.canvas_frame, text="No Video Loaded", fg="gray", bg="black", font=("Arial", 14))
        self.video_label.pack(fill="both", expand=True)
        
        self.current_image = None
        
    def show_loading(self, text="Loading..."):
        self.current_image = None
        self.video_label.configure(image="", text=text)
        
    def update_frame(self, cv2_image):
        """Convert cv2 BGR image to PIL Image and display it."""
        if cv2_image is None:
            return
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Resize logic to fit the label while maintaining aspect ratio
        target_width = self.canvas_frame.winfo_width()
        target_height = self.canvas_frame.winfo_height()
        
        # Prevent zero-size division error before window is drawn
        if target_width <= 1 or target_height <= 1:
            target_width, target_height = 800, 600
            
        img_h, img_w, _ = rgb_image.shape
        ratio = min(target_width / img_w, target_height / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
        
        rgb_image = cv2.resize(rgb_image, (new_w, new_h))
        
        # Convert to PIL and standard ImageTk
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        self.video_label.configure(image=tk_image, text="")
        # Keep a reference to prevent garbage collection
        self.current_image = tk_image
