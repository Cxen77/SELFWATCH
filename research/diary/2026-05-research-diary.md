# SELFWATCH Research Diary — May 2026

## Entry 1 — Initial Problems
Observed major ID switching during:
- crowd overlap
- pole occlusion
- sitting/standing transitions

Main issue:
nearest-visible-person rebinding.

---

## Entry 2 — Detection Optimization
Discovered RT-DETR was not actually slow.

Main bottleneck:
- PIL.Image.fromarray
- torchvision F.to_tensor

Optimization:
- replaced with torch.from_numpy zero-copy preprocessing
- moved normalization to GPU

Result:
- FPS improved from ~5 → ~18 FPS
- latency reduced from 150ms → 21ms

---

## Entry 3 — Trajectory-First Ownership
Observation:
appearance-only matching fails during overlap.

Implemented:
- direction-aware penalties
- trajectory continuity constraints
- pseudo-depth reasoning

Goal:
prioritize physical continuity over proximity.

---

## Entry 4 — Cognitive Identity States
Implemented:
- ACTIVE
- THINKING
- FROZEN
- PHANTOM

Purpose:
preserve identity ownership during ambiguity.

---

## Entry 5 — Visual Metrics
Discovered internal ID switch metrics did not match human visual perception.

Added:
- visible identity changes
- duplicate box metrics
- fragmentation tracking
- identity stability scoring
