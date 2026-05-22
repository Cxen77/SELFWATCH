import os
import json

class ForensicValidationTool:
    """
    Visual debug validation tools. Saves failure clips, before/after screenshots,
    and highlights metric-triggering frames.
    """
    def __init__(self, output_dir: str = "forensics/active"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_failure_event(self, event_type: str, frame_idx: int, metadata: dict, image_frame=None):
        """
        image_frame: numpy array of the frame
        """
        event_id = f"{event_type}_f{frame_idx}"
        event_path = os.path.join(self.output_dir, event_id)
        os.makedirs(event_path, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(event_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        # Save frame if provided
        if image_frame is not None:
            import cv2
            cv2.imwrite(os.path.join(event_path, "trigger_frame.jpg"), image_frame)
            
        # Note: Video clip saving logic would go here, utilizing a rolling buffer of frames
