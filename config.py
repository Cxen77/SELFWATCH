"""
SELFWATCH Configuration System
"""

# Detector Settings
DETECTOR_TYPE = "rtdetr"
DETECTOR_VARIANT = "nano"    # Realtime default; use medium/large for offline accuracy
DETECTOR_RESOLUTION = 384
DETECTOR_CONF_THRESH = 0.45
DETECTOR_COMPILE = False
DETECTOR_AMP = True
DETECTOR_INTERVAL = 1          # Run detector every frame — eliminates skip-frame identity churn

# StrongSORT Tracker Settings
TRACKER_TYPE = "strongsort"
TRACKER_HIGH_THRESH = 0.5
TRACKER_LOW_THRESH = 0.1
TRACKER_APPEARANCE_WEIGHT = 0.65   # Balance OSNet appearance with motion/IoU
TRACKER_MAX_COSINE_DIST = 0.30     # Tighter gate for appearance matching
TRACKER_IOU_THRESH = 0.15          # Lower gate for low-FPS/skipped-frame updates
TRACKER_MAX_LOST = 300             # Keep lost tracks alive ~16s at 18fps through occlusion
TRACKER_CONFIRM_THRESHOLD = 2     # 2 hits = fast confirmation, reduce tentative churn
TRACKER_EMBEDDING_HISTORY = 10    # Rolling embedding buffer size
TRACKER_MIN_QUALITY_SCORE = 0.4   # Min confidence to update embeddings

# ReID (OSNet) Settings
REID_MODEL = "osnet_x1_0"
REID_WEIGHTS = "weights/osnet/osnet_x1_0_msmt17.pth"
REID_EMBEDDING_DIM = 512
REID_HALF = True

# Camera Settings
DEFAULT_CAMERA = 0

# Cognitive Memory Settings
MEMORY_DEBUG_MODE = False          # Toggle debug overlay ('d' key at runtime)
MEMORY_EVENT_LOGGING = True        # Toggle JSONL event logging
MEMORY_MAX_WARM = 100              # Hard limit on warm memory entries
MEMORY_MAX_GALLERY = 5             # Max diverse embeddings per identity
MEMORY_MAX_ARCHIVE = 500           # Max archived entries (in-memory only)

# Confidence Fusion Weights (6-weight retrieval scoring)
FUSION_W_EMBEDDING = 0.55          # Embedding similarity weight (primary signal)
FUSION_W_MEMORY_CONF = 0.10       # Memory confidence weight
FUSION_W_QUALITY = 0.05           # Detection quality weight
FUSION_W_VELOCITY = 0.10          # Velocity plausibility weight
FUSION_W_GAIT = 0.10              # Gait signature similarity weight
FUSION_W_TOPOLOGY = 0.10          # Scene topology spatial prior weight
FUSION_THRESHOLD = 0.55           # Lowered: gait/topo give neutral 0.5, old 0.72 was unreachable

# Phantom Tracking Settings
PHANTOM_ENABLED = True             # Enable phantom tracking
PHANTOM_MAX_AGE = 90               # Max phantom lifetime in frames (~3s at 30fps)
PHANTOM_MATCH_THRESHOLD = 0.80    # Embedding similarity for phantom match

# Identity Contradiction Detection
CONTRADICTION_ENABLED = True       # Enable contradiction detector
CONTRADICTION_CHECK_INTERVAL = 30  # Check every N frames
CONTRADICTION_DUPLICATE_THRESH = 0.88  # Similarity threshold for duplicates
CONTRADICTION_HIJACK_THRESH = 0.55     # Similarity drop threshold for hijacks

# Cognitive Attention Priority
ATTENTION_ENABLED = True           # Enable attention-based ReID skipping
ATTENTION_HIGH_AGE = 15            # Tracks < this age = HIGH attention
ATTENTION_LOW_AGE = 90             # Tracks > this age = LOW attention
ATTENTION_RECOVERY_COOLDOWN = 20   # Frames after recovery to stay HIGH

# Scene Topology Learning
TOPOLOGY_ENABLED = True            # Enable scene topology
TOPOLOGY_GRID_SIZE = 8             # NxN zone grid
TOPOLOGY_MIN_OBSERVATIONS = 200    # Frames before topology is useful

# Gait Signature
GAIT_ENABLED = True                # Enable gait-based identity signal

# ═══════════════════════════════════════════════════════════════════
#  Multi-Camera Settings (Phase 1)
# ═══════════════════════════════════════════════════════════════════

# Cross-Camera ReID
MULTICAM_SIMILARITY_THRESHOLD = 0.70   # Cosine similarity for cross-camera match
MULTICAM_MAX_DORMANT_TIME = 300.0      # Max seconds for dormant identity survival
MULTICAM_DORMANT_DECAY_RATE = 0.998    # Exponential decay rate per tick
MULTICAM_DORMANT_MIN_CONFIDENCE = 0.05 # Min confidence before dormant expires
MULTICAM_MAX_DORMANT = 200             # Max dormant identities in memory
MULTICAM_SAME_CAMERA_MATCH = True      # Allow matching within same camera

# Event Bus
MULTICAM_EVENT_HISTORY = 10000         # Max events stored in bus
MULTICAM_TRANSITION_MAX_GAP = 120.0    # Max seconds between exit→entry pair
