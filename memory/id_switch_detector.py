import time
import math
import numpy as np

def _cosine_dist(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 1.0
    return float(1.0 - (dot / norm))

def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class UnsupervisedIDSwitchDetector:
    """
    Heuristically detects multi-object tracking ID switches without ground truth.
    Analyzes track lifecycles, spatial overlaps, and embedding continuity.
    """
    def __init__(self, metrics, time_window=1.0, space_thresh=150.0, emb_thresh=0.35):
        self.metrics = metrics
        self.time_window = time_window      # How long to remember dead tracks (seconds)
        self.space_thresh = space_thresh    # Max px distance for death+birth switch
        self.emb_thresh = emb_thresh        # Max cosine distance for same-person validation
        
        self.dead_tracks = {}               # track_id -> {box, emb, time}
        self.active_history = {}            # track_id -> deque of recent embeddings
        self.reported_switches = set()      # Prevent duplicate reporting
        self.frame_count = 0

    def tick(self, active_tracks, frame_time=None):
        """Call every frame with list of active STrack objects."""
        if frame_time is None:
            frame_time = time.perf_counter()
        self.frame_count += 1
        
        current_ids = set()
        
        # 1. Update active tracks and check for abrupt embedding shifts
        for trk in active_tracks:
            if not trk.is_confirmed:
                continue
                
            tid = trk.id
            current_ids.add(tid)
            emb = trk.embedding
            box = trk.smooth_box.tolist()
            
            if emb is None:
                continue
                
            # Abrupt Embedding Shift Detection
            if tid in self.active_history:
                hist_emb, hist_time, hist_box = self.active_history[tid]
                dt = frame_time - hist_time
                if dt < 0.5: # Only check continuous tracks
                    dist = _cosine_dist(emb, hist_emb)
                    # If embedding suddenly changes but box barely moved, it's a hijack
                    if dist > 0.4:
                        spatial_dist = math.hypot(
                            ((box[0]+box[2])-(hist_box[0]+hist_box[2]))/2.0,
                            ((box[1]+box[3])-(hist_box[1]+hist_box[3]))/2.0
                        )
                        if spatial_dist < 50.0:
                            switch_key = f"shift_{tid}_{self.frame_count}"
                            if switch_key not in self.reported_switches:
                                print(f"[ID-SWITCH] Abrupt Appearance Shift on ID {tid} (dist={dist:.2f})")
                                self.metrics.record_id_switch(tid, tid)
                                self.reported_switches.add(switch_key)
            
            self.active_history[tid] = (emb.copy(), frame_time, box)
            
            # 2. Death + Rebirth Check (Fragmentation / Failed ReID)
            # If this is a NEW track (age < 5), check if it replaced a recently dead one
            if trk.total_hits < 5:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                
                best_match = None
                best_dist = float('inf')
                
                for dead_id, dead_info in list(self.dead_tracks.items()):
                    if frame_time - dead_info["time"] > self.time_window:
                        del self.dead_tracks[dead_id]
                        continue
                        
                    dbox = dead_info["box"]
                    dcx = (dbox[0] + dbox[2]) / 2.0
                    dcy = (dbox[1] + dbox[3]) / 2.0
                    
                    spatial_dist = math.hypot(cx - dcx, cy - dcy)
                    if spatial_dist < self.space_thresh:
                        app_dist = _cosine_dist(emb, dead_info["emb"])
                        # If it looks like the dead track and is in the same place
                        if app_dist < self.emb_thresh and app_dist < best_dist:
                            best_match = dead_id
                            best_dist = app_dist
                
                if best_match is not None:
                    switch_key = f"frag_{best_match}_{tid}"
                    if switch_key not in self.reported_switches:
                        print(f"[ID-SWITCH] Track Fragmentation: Dead ID {best_match} -> New ID {tid} (app_dist={best_dist:.2f})")
                        self.metrics.record_id_switch(best_match, tid)
                        self.reported_switches.add(switch_key)
                        # Remove the dead track so it can't trigger multiple switches
                        del self.dead_tracks[best_match]

        # 3. Handle dead tracks
        # Find IDs that were in history but are no longer active
        for tid in list(self.active_history.keys()):
            if tid not in current_ids:
                emb, t, box = self.active_history.pop(tid)
                self.dead_tracks[tid] = {"emb": emb, "time": frame_time, "box": box}
                
        # 4. Crowd Intersection Swaps
        # Check active tracks that have high IoU (overlap)
        active_list = [t for t in active_tracks if t.is_confirmed and t.embedding is not None]
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                tA = active_list[i]
                tB = active_list[j]
                overlap = _iou(tA.smooth_box.tolist(), tB.smooth_box.tolist())
                if overlap > 0.4:
                    # They are heavily overlapping.
                    # Check if tA's CURRENT embedding looks more like tB's HISTORY than its own history
                    if tA.id in self.dead_tracks and tB.id in self.dead_tracks:
                        # Too complex to do safely without false positives, 
                        # but the "Abrupt Embedding Shift" check above will catch the 
                        # aftermath of an intersection swap anyway once they separate.
                        pass
