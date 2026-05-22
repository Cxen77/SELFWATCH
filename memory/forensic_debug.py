import os
import cv2
import json
import time

class ForensicDebugger:
    """
    Research/Forensic Debug Mode for SELFWATCH.
    Automatically captures failure events, saves clips from StateCache, 
    and classifies ID switches for failure taxonomy.
    """
    def __init__(self, log_dir="logs/forensic"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.event_count = 0

    def capture_id_switch(self, state_cache, current_stats, current_frame):
        """
        Triggered when an ID switch is detected in the pipeline.
        Saves a 5-second context clip (150 frames) and detailed JSON metadata.
        """
        self.event_count += 1
        timestamp = int(time.time())
        prefix = os.path.join(self.log_dir, f"fail_{timestamp}_{self.event_count}")
        
        # 1. Save exact failure frame image
        cv2.imwrite(f"{prefix}_frame.jpg", current_frame)
        
        # 2. Extract 5-second clip from cache (e.g. 150 frames @ 30fps)
        clip_frames = []
        end_idx = state_cache.current_index
        start_idx = max(0, end_idx - 150)
        
        for i in range(start_idx, end_idx + 1):
            f, _ = state_cache.get_frame(i)
            if f is not None:
                clip_frames.append(f)
                
        if clip_frames:
            h, w = clip_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f"{prefix}_clip.mp4", fourcc, 30.0, (w, h))
            for f in clip_frames:
                out.write(f)
            out.release()

        # 3. Classify Failure
        taxonomy = self._classify_failure(current_stats)

        # 4. Save JSON Logs
        metadata = {
            "timestamp": timestamp,
            "raw_frame_index": current_stats.get("raw_frame_index"),
            "id_switches": current_stats.get("id_switches", []),
            "active_tracks": current_stats.get("active_tracks", 0),
            "frozen_gids": current_stats.get("frozen_gids", []),
            "suppression_regions": current_stats.get("suppress_regions", []),
            "taxonomy": taxonomy,
            "track_states": current_stats.get("track_states", {})
        }
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
                
        with open(f"{prefix}_meta.json", "w") as f:
            json.dump(metadata, f, indent=4, cls=NumpyEncoder)
            
        print(f"\n[FORENSIC] Captured ID Switch Event -> {prefix}")
        print(f"[FORENSIC] Classification: {taxonomy}\n")

    def _classify_failure(self, stats):
        """Analyze state to classify the type of ID switch."""
        frozen = stats.get("frozen_gids", [])
        switches = stats.get("id_switches", [])
        
        if not switches:
            return "unknown"
            
        # Simplistic taxonomy rules
        if frozen:
            return "crossing_failure"
        if stats.get("suppress_regions"):
            return "suppression_expiry"
            
        return "track_fragmentation"
