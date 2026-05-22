import cv2
import numpy as np
from collections import deque

class StateCache:
    """
    Caches processed frames to support timeline scrubbing in the UI.
    Uses in-memory JPEG compression to reduce RAM usage.
    """
    def __init__(self, max_frames=2000):
        self.max_frames = max_frames
        self.frames = []
        self.metadata = []
        self.current_index = -1
        
    def append(self, frame, meta):
        """Append a new processed frame to the frontier."""
        # Compress to JPEG to save RAM (takes ~1-2ms, saves 90% memory)
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # If we scrubbed backward and then play, we must truncate the future
        if self.current_index < len(self.frames) - 1:
            self.frames = self.frames[:self.current_index + 1]
            self.metadata = self.metadata[:self.current_index + 1]
            
        self.frames.append(encoded.tobytes())
        self.metadata.append(meta)
        
        # Enforce max limit
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
            self.metadata.pop(0)
        else:
            self.current_index += 1
            
    def get_frame(self, index):
        """Retrieve decoded frame and metadata at index."""
        if index < 0 or index >= len(self.frames):
            return None, None
            
        encoded = self.frames[index]
        meta = self.metadata[index]
        
        frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        self.current_index = index
        return frame, meta
        
    def clear(self):
        self.frames.clear()
        self.metadata.clear()
        self.current_index = -1
        
    @property
    def total_frames(self):
        return len(self.frames)
