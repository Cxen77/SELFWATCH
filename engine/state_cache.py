import cv2
import numpy as np
from collections import deque

class StateCache:
    """
    Caches processed frames to support timeline scrubbing in the UI.
    Uses in-memory JPEG compression to reduce RAM usage.
    
    Memory-bounded: uses deque(maxlen) to enforce hard frame limit.
    At 300 frames × 2 cameras × ~30KB JPEG each ≈ 18MB total.
    """
    def __init__(self, max_frames=300):
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        self.metadata = deque(maxlen=max_frames)
        self.current_index = -1
        
    def append(self, frames_list, meta_list):
        """Append a new set of processed frames to the frontier."""
        encoded_list = []
        for frame in frames_list:
            if frame is not None:
                # Encode at lower quality for memory savings
                _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                encoded_list.append(encoded.tobytes())
            else:
                encoded_list.append(None)
                
        # If we scrubbed backward and then play, truncate the future
        if self.current_index < len(self.frames) - 1 and self.current_index >= 0:
            # Convert deque to list, truncate, convert back
            frames_list_tmp = list(self.frames)[:self.current_index + 1]
            meta_list_tmp = list(self.metadata)[:self.current_index + 1]
            self.frames = deque(frames_list_tmp, maxlen=self.max_frames)
            self.metadata = deque(meta_list_tmp, maxlen=self.max_frames)
            
        self.frames.append(encoded_list)
        self.metadata.append(meta_list)
        
        # deque(maxlen) handles eviction automatically
        self.current_index = len(self.frames) - 1
            
    def get_frame(self, index):
        """Retrieve decoded frames and metadata at index."""
        if index < 0 or index >= len(self.frames):
            return None, None
            
        encoded_list = self.frames[index]
        meta_list = self.metadata[index]
        
        decoded_list = []
        for encoded in encoded_list:
            if encoded is not None:
                frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                decoded_list.append(frame)
            else:
                decoded_list.append(None)
                
        self.current_index = index
        return decoded_list, meta_list
        
    def clear(self):
        self.frames.clear()
        self.metadata.clear()
        self.current_index = -1
        
    @property
    def total_frames(self):
        return len(self.frames)
