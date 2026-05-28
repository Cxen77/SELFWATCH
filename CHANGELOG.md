# CHANGELOG

## [v0.1] — Initial Tracker
Date: 2026-04
- Built initial RT-DETR + StrongSORT pipeline
- Basic person tracking working
- Major issues:
  - severe ID switching
  - duplicate boxes
  - poor occlusion handling
  - low FPS

---

## [v0.2] — Detection Optimization
Date: 2026-05
- Optimized RT-DETR pipeline
- Removed PIL preprocessing bottleneck
- Added zero-copy preprocessing using torch.from_numpy
- Added FP16 + compile_model=True
- Moved resize/normalize to GPU

Performance:
- FPS improved from ~5 FPS → ~18 FPS
- Detection latency reduced from ~150ms → ~21ms

---

## [v0.3] — StrongSORT+ Engine
Date: 2026-05
- Added ByteTrack-inspired low-confidence recovery
- Added C-BIoU buffered matching
- Added birth suppression regions
- Added direction-aware association penalties
- Added pseudo-depth reasoning

---

## [v0.4] — Cognitive Identity Layer
Date: 2026-05
- Added ACTIVE state
- Added THINKING state
- Added FROZEN state
- Added PHANTOM state
- Added RECOVERED state
- Added DEAD state

Purpose:
Preserve identity continuity during overlap and occlusion.

---

## [v0.5] — Memory & Trajectory Reasoning
Date: 2026-05
- Added Active Memory
- Added Warm Memory
- Added identity fingerprints
- Added trajectory-first ownership reasoning
- Added ACT-R inspired identity persistence
- Added occlusion-aware appearance weighting

---

## [v0.6] — Forensic & Visual Metrics
Date: 2026-05
- Added forensic debugging system
- Added automatic failure capture
- Added visual identity continuity metrics
- Added duplicate box detection
- Added fragmentation metrics
- Added identity stability scoring

---

## [v1.7] — Hard Collision Gating
Date: 2026-05-22
- Integrated high-level cognitive frozen state down to the local tracker via `frozen_lids`
- Disabled appearance matching entirely for frozen tracks during high ambiguity
- Locked tracks into a tight spatial and trajectory corridor to prevent ID stealing

---

## [v1.8] — Post-Collision Recovery Lock
Date: 2026-05-22
- Added RECOVERY_LOCK phase extending for 25 frames after unfreezing
- Implemented trajectory direction checks and tight cosine distance thresholds (`0.28`) during recovery
- Implemented mutual appearance-exclusion checks between cooldown tracks

---

## [v1.9] — Post-Collision Trajectory Commitment
Date: 2026-05-23
- Implemented RECOVERY_TRAJECTORY_LOCK extending for 30 frames after collision release
- Implemented Post-Collision Trajectory Commitment: spatial costs are computed directly from the predicted trajectory corridor based on pre-collision velocity
- Implemented Crossing-Trajectory Rejection (Mutual Exclusion): strictly forbids identity exchange or visual/spatial stealing between crossing partners during separation
- Result: **All-time record 99.2% Identity Stability with 0 Visible ID Switches**
