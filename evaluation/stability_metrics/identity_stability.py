class IdentityStabilityScore:
    """
    Combines visual ID switches, duplicate boxes, teleportation, fragmentation,
    and occlusion recovery quality into a human-perceived stability score.
    """
    def __init__(self):
        pass
        
    def compute_score(self, visible_switches: int, duplicates: int, 
                      teleportations: int, fragmentations: int, 
                      total_frames: int, active_tracks: int) -> float:
        """
        Calculates a score between 0.0 and 1.0 where 1.0 is perfectly stable.
        """
        # A simple penalty-based heuristic for human perception
        # Weights reflect how jarring the error is to a human viewer
        penalty = (visible_switches * 2.0) + (duplicates * 0.5) + \
                  (teleportations * 3.0) + (fragmentations * 1.0)
                  
        normalization_factor = total_frames * max(1, active_tracks * 0.1)
        raw_score = 1.0 - (penalty / normalization_factor)
        
        return max(0.0, min(1.0, raw_score))
