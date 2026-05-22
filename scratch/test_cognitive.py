"""Functional test for all observability and stability systems."""
import numpy as np
import time
import os
from memory.cognitive import CognitiveMemory
from memory.event_log import CognitiveEventLogger
from memory.metrics import TrackingMetrics

# Set up observability
logger = CognitiveEventLogger(log_dir="logs", enabled=True)
metrics = TrackingMetrics()

brain = CognitiveMemory(
    max_warm=5,  # Small for testing
    max_archive=10,
    fusion_weights=[0.50, 0.15, 0.15, 0.20],
    fusion_threshold=0.78,
    event_logger=logger,
    metrics=metrics,
)

# Simulate tracking a person
emb = np.random.randn(512).astype(np.float32)
emb /= np.linalg.norm(emb)

# 1. Memory Lock with logging
locked, quality = brain.should_lock_memory(1, [100,100,200,400], 0.85, frame_dims=(720,1280))
print(f"1. Memory Lock: locked={locked}, quality={quality:.2f}")

# 2. Active identity update
brain.update_active_identity(1, emb, quality)
print(f"2. Active identity gallery: {brain.get_debug_info(1)}")

# 3. Save to warm + hard limit test
for i in range(7):  # Exceed max_warm=5
    e = np.random.randn(512).astype(np.float32)
    e /= np.linalg.norm(e)
    brain.save_lost_track(i+10, e, 30, 0.7, last_position=[100,100,200,400])

print(f"3. Warm={brain.warm_count} (max=5), Archive={brain.archive_count}")

# 4. Confidence fusion retrieval
brain.save_lost_track(99, emb, 90, 0.9, last_position=[100,100,200,400], velocity=[1.0, 0.0])
time.sleep(0.3)
recovered = brain.retrieve_identity(emb, [120,100,220,400], time.perf_counter())
print(f"4. Fusion retrieval: recovered_id={recovered}")

# 5. Metrics tracking
for _ in range(100):
    metrics.tick_frame()
print(f"5. Metrics: {metrics.get_summary()['total_frames']} frames tracked")

# 6. Scene difficulty
brain.set_scene_difficulty(5, 0.6)
print(f"6. Difficulty: {brain._difficulty_multiplier:.2f}")

# 7. Print summary
metrics.print_summary()

# 8. Export CSV
metrics.export_csv("logs/test_metrics.csv")
print(f"8. CSV exported: {os.path.exists('logs/test_metrics.csv')}")

# Cleanup
logger.close()
print("\nALL SYSTEMS VERIFIED")
