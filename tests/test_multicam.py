"""Quick functional test for multicam Phase 1 infrastructure."""
import time
import numpy as np
from multicam.global_registry import GlobalMultiCameraIdentityManager
from multicam.events import CameraEventBus, CameraEvent, EventType
from multicam.cross_camera_reid import CrossCameraReIDMatcher


def run_tests():
    passed = 0
    total = 0

    # Test 1: Global Registry - ID allocation
    total += 1
    reg = GlobalMultiCameraIdentityManager()
    emb1 = np.random.randn(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    emb2 = np.random.randn(512).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)

    gid1 = reg.register_local_track(0, 10, embedding=emb1)
    gid2 = reg.register_local_track(1, 20, embedding=emb2)
    assert gid1 != gid2, "Different tracks should get different GIDs"
    print(f"Test 1 - ID allocation: GID1={gid1}, GID2={gid2} ... OK")
    passed += 1

    # Test 2: Re-registration returns same GID
    total += 1
    gid1b = reg.register_local_track(0, 10)
    assert gid1 == gid1b, "Re-registration should return same GID"
    print(f"Test 2 - Re-registration: same GID ... OK")
    passed += 1

    # Test 3: Dormant identity
    total += 1
    reg.unregister_local_track(0, 10)
    reg.move_to_dormant(gid1, 0)
    dormant = reg.get_dormant_identities()
    assert gid1 in dormant, "GID1 should be dormant"
    print(f"Test 3 - Dormant: {len(dormant)} entries ... OK")
    passed += 1

    # Test 4: Event Bus
    total += 1
    bus = CameraEventBus()
    bus.publish(CameraEvent(EventType.EXIT, gid1, 0, time.time(), 50))
    bus.publish(CameraEvent(EventType.ENTER, gid1, 1, time.time(), 60))
    assert bus.total_events == 2
    print(f"Test 4 - Events: {bus.get_summary()} ... OK")
    passed += 1

    # Test 5: Cross-Camera Match (similar embedding)
    total += 1
    matcher = CrossCameraReIDMatcher(reg, bus, similarity_threshold=0.70)
    emb = dormant[gid1].stable_embedding.copy()
    emb += np.random.randn(512).astype(np.float32) * 0.01  # tiny noise
    emb /= np.linalg.norm(emb)
    result = matcher.attempt_match(1, 99, emb, frame_index=100)
    assert result == gid1, f"Match failed: got {result}, expected {gid1}"
    print(f"Test 5 - Cross-camera match: GID={result} ... OK")
    passed += 1

    # Test 6: No match for different person
    total += 1
    diff_emb = np.random.randn(512).astype(np.float32)
    diff_emb /= np.linalg.norm(diff_emb)
    result2 = matcher.attempt_match(1, 100, diff_emb, frame_index=101)
    assert result2 is None, f"Should not match: got {result2}"
    print(f"Test 6 - No match for different person ... OK")
    passed += 1

    # Test 7: Registry stats
    total += 1
    stats = reg.get_stats()
    assert stats["total_global_ids_created"] >= 2
    assert stats["total_cross_camera_matches"] >= 1
    print(f"Test 7 - Stats: {stats} ... OK")
    passed += 1

    # Test 8: Memory entry query
    total += 1
    entry = reg.get_memory_entry(gid1)
    assert entry is not None
    # After cross-camera match, the identity is registered again
    print(f"Test 8 - Memory entry: state={entry.state}, obs={entry.total_observations} ... OK")
    passed += 1

    # Test 9: Dormant decay to expiration
    total += 1
    # Create a fresh dormant identity for decay test
    reg2 = GlobalMultiCameraIdentityManager()
    emb3 = np.random.randn(512).astype(np.float32)
    emb3 /= np.linalg.norm(emb3)
    gid3 = reg2.register_local_track(0, 30, embedding=emb3)
    reg2.unregister_local_track(0, 30)
    reg2.move_to_dormant(gid3, 0)
    assert len(reg2.get_dormant_identities()) == 1

    for _ in range(5000):
        reg2.decay_dormant()
    dormant_after = reg2.get_dormant_identities()
    assert len(dormant_after) == 0, f"Expected 0 dormant, got {len(dormant_after)}"
    print(f"Test 9 - Dormant expiration: 0 remaining ... OK")
    passed += 1

    print(f"\n{'='*50}")
    print(f"  All {passed}/{total} tests PASSED!")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_tests()
