"""Test script for temporal smoothing, soft-lock, and area-percentage edge."""
import numpy as np
from memory.cognitive import CognitiveMemory
from memory.metrics import TrackingMetrics

print("=" * 60)
print("  SELFWATCH - Architecture Fixes Validation")
print("=" * 60)

metrics = TrackingMetrics()
brain = CognitiveMemory(metrics=metrics)

# 1. Normal frame (No lock)
is_locked, qual, lock_type = brain.should_lock_memory(
    track_id=1, 
    bbox=[100, 100, 200, 200], 
    confidence=0.9, 
    frame_dims=(1080, 1920)
)
assert not is_locked
assert lock_type is None
assert qual > 0.8
print("  [OK] Normal frame: Not locked")

# 2. Near-edge frame (x1=2, but box is 100px wide -> 98% visible) -> NO LOCK
is_locked, qual, lock_type = brain.should_lock_memory(
    track_id=2, 
    bbox=[2, 100, 102, 300],  # Near edge but 98% visible
    confidence=0.9, 
    frame_dims=(1080, 1920)
)
assert not is_locked, f"Should NOT lock near-edge, got lock_type={lock_type}"
assert lock_type is None, f"Should be None, got {lock_type}"
print("  [OK] Near-edge (98% visible): Not locked — FIXED!")

# 3. Truly truncated bbox (50% visible) -> Should start soft lock
is_locked, qual, lock_type = brain.should_lock_memory(
    track_id=3, 
    bbox=[-50, 100, 50, 300],  # 50% outside frame
    confidence=0.9, 
    frame_dims=(1080, 1920)
)
assert not is_locked, "Should not hard lock on first bad frame"
assert lock_type == "soft_lock", f"Expected soft_lock, got {lock_type}"
print("  [OK] Truncated frame 1 (50% visible): Soft Lock")

# 4. Second truncated frame
is_locked, qual, lock_type = brain.should_lock_memory(
    track_id=3, bbox=[-50, 100, 50, 300], confidence=0.9, frame_dims=(1080, 1920))
assert lock_type == "soft_lock"
print("  [OK] Truncated frame 2: Soft Lock continues")

# 5. Third truncated frame -> HARD LOCK
is_locked, qual, lock_type = brain.should_lock_memory(
    track_id=3, bbox=[-50, 100, 50, 300], confidence=0.9, frame_dims=(1080, 1920))
assert is_locked, "Should hard lock on third bad frame"
assert lock_type == "hard_lock"
print("  [OK] Truncated frame 3: Hard Lock triggered")

# 6. Recovery
is_locked, qual, lock_type = brain.should_lock_memory(
    track_id=3, bbox=[100, 100, 200, 300], confidence=0.9, frame_dims=(1080, 1920))
assert not is_locked
assert lock_type is None
print("  [OK] Recovery: Locks cleared")

# 7. Test 6-weight fusion default
assert len(brain._fusion_weights) == 6, f"Expected 6 weights, got {len(brain._fusion_weights)}"
print(f"  [OK] 6-weight fusion: {brain._fusion_weights}")

# 8. Test save_lost_track with gait
fake_emb = np.random.randn(512).astype(np.float32)
fake_emb /= np.linalg.norm(fake_emb)
fake_gait = np.array([1.5, 0.3, 0.7, 0.02, 0.03, 5.0, 0.4, 0.35], dtype=np.float32)
brain.save_lost_track(
    track_id=99, final_embedding=fake_emb, duration_frames=100,
    quality_score=0.85, last_position=[100, 100, 200, 400],
    velocity=[2.0, 0.5], gait_signature=fake_gait,
)
assert 99 in brain.warm_memory
stored_gait = brain.warm_memory[99].get("gait_signature")
assert stored_gait is not None, "Gait signature should be stored in warm memory"
assert np.allclose(stored_gait, fake_gait)
print("  [OK] Warm memory stores gait signature")

# 9. Test retrieval finds the identity
test_emb = fake_emb + np.random.randn(512).astype(np.float32) * 0.01
test_emb /= np.linalg.norm(test_emb)
import time
# Age the warm memory entry so it passes the dt>0.2 temporal constraint
brain.warm_memory[99]["timestamp"] = time.perf_counter() - 1.0
recovered = brain.retrieve_identity(
    new_embedding=test_emb,
    current_position=[110, 110, 210, 410],
    current_time=time.perf_counter(),
)
assert recovered == 99, f"Should recover ID 99, got {recovered}"
print("  [OK] Retrieval engine works with 6-weight fusion")

print(f"  Metrics -> Hard: {metrics.hard_locks}, Soft: {metrics.soft_locks}")
print(f"  Lock Reasons -> {metrics.lock_reasons}")
print("=" * 60)
print("  ALL TESTS PASSED")
