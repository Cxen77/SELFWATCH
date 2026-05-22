import time
import math
import numpy as np

class CognitiveMemory:
    def __init__(self, base_decay_rate=0.05, archive_threshold=0.3):
        self.warm_memory = {} # Stores identities waiting to reappear
        self.base_decay_rate = base_decay_rate
        self.archive_threshold = archive_threshold

    def save_lost_track(self, track_id, embedding, duration, quality_score, last_box=None):
        # 1. Calculate Importance (More visible/higher quality = more important)
        # Assuming duration is in frames. 30 frames = 1.0 importance boost.
        importance = (duration / 30.0) + quality_score 
        
        # 2. Push to Warm Memory with 100% confidence to start
        self.warm_memory[track_id] = {
            "embedding": embedding,
            "importance": importance,
            "confidence": 1.0,  # Starts at 1.0 (100%)
            "last_time": time.perf_counter(),
            "last_box": last_box
        }
        print(f"🧠 Track {track_id} saved to Warm Memory! Importance: {importance:.2f}")

    def retrieve_identity(self, new_embedding, new_box, current_time, match_threshold=0.85):
        """Searches the graveyard for a visual match with spatial and temporal constraints."""
        if not self.warm_memory:
            return None # Graveyard is empty

        best_match_id = None
        highest_similarity = 0.0

        new_cx = (new_box[0] + new_box[2]) / 2
        new_cy = (new_box[1] + new_box[3]) / 2

        # Search through all sleeping memories
        for track_id, mem in self.warm_memory.items():
            dt = current_time - mem["last_time"]
            
            # Temporal constraint: Prevent immediate rapid-fire resurrects (could be ghost boxes)
            if dt < 0.2:
                continue
                
            # Spatial constraint: plausible movement distance
            if mem.get("last_box") is not None:
                old_box = mem["last_box"]
                old_cx = (old_box[0] + old_box[2]) / 2
                old_cy = (old_box[1] + old_box[3]) / 2
                dist = math.hypot(new_cx - old_cx, new_cy - old_cy)
                
                # Allow 1500 pixels per second, max 3000px limit
                max_dist = min(3000, 1500 * dt + 200) # +200 base leeway
                if dist > max_dist:
                    continue

            # Cosine similarity between new person and sleeping memory
            similarity = np.dot(new_embedding, mem["embedding"]) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(mem["embedding"])
            )
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_id = track_id

        # If it's a strong match, RESURRECT THEM
        if highest_similarity >= match_threshold:
            print(f"⚡ RESURRECTION! Restored ID {best_match_id} (Sim: {highest_similarity:.2f}, Gap: {current_time - self.warm_memory[best_match_id]['last_time']:.1f}s)")
            
            # Give them a reinforcement spike (they survived!)
            self.warm_memory[best_match_id]["confidence"] = 1.0 
            self.warm_memory[best_match_id]["importance"] += 0.5 
            
            return best_match_id
            
        return None # Truly a new person

    def update_and_decay(self):
        current_time = time.perf_counter()
        deleted_keys = []

        for track_id, mem in self.warm_memory.items():
            # Calculate time passed since last update
            dt = current_time - mem["last_time"]
            mem["last_time"] = current_time
            
            # The Magic Math: Important memories decay slower
            effective_decay = self.base_decay_rate / (1.0 + mem["importance"])
            
            # Apply Exponential Decay: C_t = C_prev * e^(-lambda * dt)
            mem["confidence"] = mem["confidence"] * math.exp(-effective_decay * dt)

            # If confidence drops too low, flag for Archive/Deletion
            if mem["confidence"] < self.archive_threshold:
                deleted_keys.append(track_id)

        # Remove dead memories
        for track_id in deleted_keys:
            print(f"💀 Track {track_id} decayed completely. Pushing to Archive/Deleting.")
            del self.warm_memory[track_id]
