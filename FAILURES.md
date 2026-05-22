# Known Failures

## 1. Opposite-Direction Rebinding
Problem:
global ID attaches to another person moving in opposite direction.

Likely Cause:
proximity weighting still too strong.

Current Fixes:
- direction-aware penalties
- trajectory continuity constraints
- pseudo-depth gating

---

## 2. Duplicate Boxes
Problem:
old THINKING track and new local track both visible.

Common During:
- sitting → standing
- fragmentation
- delayed merge

---

## 3. Pose Change Fragmentation
Problem:
tracker creates new local track after drastic appearance change.

Common During:
- standing up
- turning
- partial occlusion
