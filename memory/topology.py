"""
SELFWATCH - Scene Topology Learning

Learns the spatial structure of the camera's view:
  - WHERE do people enter the frame?
  - WHERE do people exit the frame?
  - WHAT paths do they take? (transition probabilities)

This learned knowledge acts as a SPATIAL PRIOR during identity retrieval.
If a person was lost in zone A and reappears in zone B, the system checks
how likely the A->B transition is. High probability = boost match confidence.
Low probability = suppress match.

The system learns automatically from observed tracking data. No calibration
or manual setup required. After ~200 frames it starts producing useful priors.

Cost: Counter increments per frame + one matrix lookup during retrieval.
"""

import numpy as np


class SceneTopology:
    """
    Learns spatial structure from tracking data.

    Divides the frame into an NxN grid. Tracks three things:
      1. Entry heatmap: where new tracks first appear
      2. Exit heatmap: where tracks last appear before loss
      3. Transition matrix: probability of moving from zone i to zone j

    Args:
        grid_size:       NxN grid divisions (8 = 64 zones).
        min_observations: Minimum observations before topology is "ready".
        event_logger:    CognitiveEventLogger or None.
    """

    def __init__(self, grid_size=8, min_observations=200,
                 event_logger=None):
        self.grid_size = grid_size
        self.min_observations = min_observations
        self._logger = event_logger
        self._frame_dims = None  # (height, width), set on first call

        n = grid_size * grid_size

        # Heatmaps
        self._entry_counts = np.zeros(n, dtype=np.float32)
        self._exit_counts = np.zeros(n, dtype=np.float32)

        # Transition matrix: transitions[from_zone][to_zone] = count
        self._transitions = np.zeros((n, n), dtype=np.float32)

        # Per-track zone history
        self._track_zones = {}     # track_id -> last_zone_index
        self._total_observations = 0
        self._is_ready = False

    def _pos_to_zone(self, position):
        """Convert a bbox center to a zone index."""
        if self._frame_dims is None:
            return 0

        cx = (position[0] + position[2]) / 2.0
        cy = (position[1] + position[3]) / 2.0

        fh, fw = self._frame_dims
        col = int(min(cx / fw * self.grid_size, self.grid_size - 1))
        row = int(min(cy / fh * self.grid_size, self.grid_size - 1))
        col = max(0, col)
        row = max(0, row)

        return row * self.grid_size + col

    def set_frame_dims(self, frame_shape):
        """Set frame dimensions (call once or when resolution changes)."""
        self._frame_dims = (frame_shape[0], frame_shape[1])

    def record_entry(self, track_id, position):
        """Record where a new track first appeared."""
        zone = self._pos_to_zone(position)
        self._entry_counts[zone] += 1
        self._track_zones[track_id] = zone
        self._total_observations += 1
        self._check_ready()

    def record_exit(self, track_id, position):
        """Record where a track was last seen before loss."""
        zone = self._pos_to_zone(position)
        self._exit_counts[zone] += 1
        self._track_zones.pop(track_id, None)
        self._total_observations += 1
        self._check_ready()

    def update_position(self, track_id, position):
        """Update current zone for a tracked person (call each frame)."""
        new_zone = self._pos_to_zone(position)
        old_zone = self._track_zones.get(track_id)

        if old_zone is not None and old_zone != new_zone:
            self._transitions[old_zone][new_zone] += 1
            self._total_observations += 1

        self._track_zones[track_id] = new_zone
        self._check_ready()

    def get_transition_probability(self, from_position, to_position):
        """
        Get the learned probability of transitioning between two positions.

        Returns a value between 0.0 and 1.0. Returns 0.5 (neutral) if the
        system hasn't learned enough yet.

        Args:
            from_position: [x1, y1, x2, y2] where the person was lost.
            to_position:   [x1, y1, x2, y2] where the person reappeared.
        """
        if not self._is_ready:
            return 0.5  # Neutral prior

        from_zone = self._pos_to_zone(from_position)
        to_zone = self._pos_to_zone(to_position)

        # Row-normalize transitions to get probabilities
        row_sum = self._transitions[from_zone].sum()
        if row_sum < 1.0:
            return 0.5  # Not enough data for this zone

        prob = self._transitions[from_zone][to_zone] / row_sum
        return float(prob)

    def get_spatial_prior(self, from_position, to_position):
        """
        Compute a spatial plausibility score for retrieval fusion.

        Combines transition probability with entry/exit zone knowledge.
        Returns 0.0 (impossible) to 1.0 (highly likely).

        This value can be used as an additional weight in the confidence
        fusion scoring: S = w1*E + w2*M + w3*Q + w4*V + w5*TOPO
        """
        if not self._is_ready:
            return 0.5

        trans_prob = self.get_transition_probability(from_position, to_position)

        # Also check if the reappearance zone is a known entry zone
        to_zone = self._pos_to_zone(to_position)
        entry_total = self._entry_counts.sum()
        if entry_total > 0:
            entry_prob = self._entry_counts[to_zone] / entry_total
        else:
            entry_prob = 0.5

        # Blend: transition probability is primary, entry knowledge is secondary
        prior = 0.7 * trans_prob + 0.3 * min(1.0, entry_prob * self.grid_size * self.grid_size)

        return max(0.0, min(1.0, prior))

    def _check_ready(self):
        """Check if enough data has been collected."""
        if not self._is_ready and self._total_observations >= self.min_observations:
            self._is_ready = True
            if self._logger:
                self._logger.log("topology_ready",
                                 observations=self._total_observations)

    @property
    def is_ready(self):
        return self._is_ready

    def get_entry_zones(self, top_n=3):
        """Return the top-N most common entry zones."""
        indices = np.argsort(self._entry_counts)[::-1][:top_n]
        return [(int(i), float(self._entry_counts[i])) for i in indices
                if self._entry_counts[i] > 0]

    def get_exit_zones(self, top_n=3):
        """Return the top-N most common exit zones."""
        indices = np.argsort(self._exit_counts)[::-1][:top_n]
        return [(int(i), float(self._exit_counts[i])) for i in indices
                if self._exit_counts[i] > 0]

    def clear_track(self, track_id):
        """Clean up per-track state."""
        self._track_zones.pop(track_id, None)
