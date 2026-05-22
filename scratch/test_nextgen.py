"""Functional test for all 5 next-gen cognitive systems."""
import numpy as np
import time
from memory.phantom import PhantomTracker
from memory.contradiction import ContradictionDetector
from memory.attention import CognitiveAttention, TIER_HIGH, TIER_NORMAL, TIER_LOW
from memory.topology import SceneTopology
from memory.gait import GaitSignature

print("=" * 60)
print("  SELFWATCH - Next-Gen Cognitive Systems Test")
print("=" * 60)

# ── 1. PHANTOM TRACKING ──────────────────────────────────────────
print("\n[1] Phantom Tracking")
pt = PhantomTracker(max_phantom_age=90, match_threshold=0.80)

emb = np.random.randn(512).astype(np.float32)
emb /= np.linalg.norm(emb)
gallery = [emb.copy()]

pt.spawn(42, emb, [100, 100, 200, 400], [5.0, 0.0], importance=2.0, gallery=gallery)
assert pt.count == 1, "Phantom should exist"

# Tick 10 frames
for _ in range(10):
    pt.tick()
assert pt.count == 1, "Phantom should still be alive"

# Try matching with same embedding near predicted position
# After 10 frames at vx=5, cx moved ~50px (with 0.95 friction decay)
match = pt.try_match(emb, [140, 100, 240, 400])
assert match is not None, "Should match phantom"
assert match.track_id == 42, f"Should be ID 42, got {match.track_id}"
pt.remove(42)
assert pt.count == 0, "Phantom should be removed after match"

# Test expiry
pt.spawn(99, emb, [100, 100, 200, 400], [1.0, 0.0])
for _ in range(100):
    pt.tick()
assert pt.count == 0, "Phantom should have expired"
print("  OK - spawn, predict, match, expire all working")

# ── 2. IDENTITY CONTRADICTION ────────────────────────────────────
print("\n[2] Identity Contradiction Detector")

class FakeTrack:
    def __init__(self, tid, emb, confirmed=True, hits=50):
        self.id = tid
        self.embedding = emb.copy()
        self.is_confirmed = confirmed
        self.time_since_update = 0
        self.total_hits = hits
        self.age = hits

cd = ContradictionDetector(check_interval=1)

emb_a = np.random.randn(512).astype(np.float32)
emb_a /= np.linalg.norm(emb_a)
emb_b = emb_a.copy()  # Identical embedding = duplicate

tracks = [FakeTrack(1, emb_a), FakeTrack(2, emb_b)]

result = cd.tick(tracks)
assert len(result["duplicates"]) > 0, "Should detect duplicate"
print(f"  Duplicate detected: {result['duplicates'][0]}")

# Hijack test: build history then shift embedding
emb_c = np.random.randn(512).astype(np.float32)
emb_c /= np.linalg.norm(emb_c)
cd2 = ContradictionDetector(check_interval=1)
track_c = FakeTrack(10, emb_c)
for _ in range(8):
    cd2.tick([track_c])
# Now shift embedding dramatically
track_c.embedding = np.random.randn(512).astype(np.float32)
track_c.embedding /= np.linalg.norm(track_c.embedding)
result2 = cd2.tick([track_c])
assert len(result2["hijacks"]) > 0, "Should detect hijack"
print(f"  Hijack detected: {result2['hijacks'][0]}")

# ── 3. COGNITIVE ATTENTION ───────────────────────────────────────
print("\n[3] Cognitive Attention Priority")
att = CognitiveAttention(high_age_thresh=15, low_age_thresh=90)

att.update_tier(1, track_age=5, is_confirmed=True, frame_count=0)
assert att.get_tier(1) == TIER_HIGH, "Young track should be HIGH"

att.update_tier(2, track_age=50, is_confirmed=True, frame_count=0)
assert att.get_tier(2) == TIER_NORMAL, "Mid-age track should be NORMAL"

att.update_tier(3, track_age=120, is_confirmed=True, frame_count=0)
assert att.get_tier(3) == TIER_LOW, "Old track should be LOW"

# ReID skip logic
assert att.should_extract_reid(1, 0) == True, "HIGH always extracts"
assert att.should_extract_reid(3, 0) == False, "LOW should skip"

stats = att.get_stats()
print(f"  Tiers: H={stats['HIGH']} N={stats['NORMAL']} L={stats['LOW']}")
print("  OK - attention tiers and ReID skipping working")

# ── 4. SCENE TOPOLOGY ───────────────────────────────────────────
print("\n[4] Scene Topology Learning")
topo = SceneTopology(grid_size=4, min_observations=10)
topo.set_frame_dims((480, 640))

# Simulate entries at top-left
for i in range(5):
    topo.record_entry(i, [10, 10, 50, 100])
# Simulate exits at bottom-right
for i in range(5):
    topo.record_exit(i, [580, 400, 630, 470])

# Simulate transitions
for _ in range(5):
    topo.update_position(100, [10, 10, 50, 100])
    topo.update_position(100, [300, 200, 350, 300])
    topo.update_position(100, [580, 400, 630, 470])

assert topo.is_ready, "Should be ready after enough observations"
prior = topo.get_spatial_prior([10, 10, 50, 100], [580, 400, 630, 470])
print(f"  Topology ready, spatial prior = {prior:.3f}")
print(f"  Entry zones: {topo.get_entry_zones(2)}")
print(f"  Exit zones: {topo.get_exit_zones(2)}")

# ── 5. GAIT SIGNATURE ───────────────────────────────────────────
print("\n[5] Gait Signature")
gait = GaitSignature(fps_estimate=15)

# Simulate walking with head oscillation
for frame in range(45):
    # Simulated bbox with periodic height oscillation (walking)
    osc = 5 * np.sin(2 * np.pi * 2.0 * frame / 15)  # 2Hz oscillation
    x1 = 100 + frame * 3  # Moving right
    bbox = [x1, 100 - osc, x1 + 60, 350 + osc]
    gait.update(1, bbox)

sig = gait.get_signature(1)
assert sig is not None, "Should have signature after 45 frames"
print(f"  Gait vector (8-dim): [{', '.join(f'{v:.3f}' for v in sig)}]")

# Compare with self (should be 1.0)
self_sim = GaitSignature.compare(sig, sig)
print(f"  Self-similarity: {self_sim:.3f}")

# Compare with different person (different stride)
for frame in range(45):
    osc2 = 3 * np.sin(2 * np.pi * 1.5 * frame / 15)  # Different frequency
    bbox2 = [200 + frame * 2, 120 - osc2, 260 + frame * 2, 370 + osc2]
    gait.update(2, bbox2)
sig2 = gait.get_signature(2)
cross_sim = GaitSignature.compare(sig, sig2)
print(f"  Cross-similarity (different person): {cross_sim:.3f}")
assert cross_sim < self_sim, "Different gaits should be less similar"

print("\n" + "=" * 60)
print("  ALL 5 SYSTEMS VERIFIED")
print("=" * 60)
