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
