# SELFWATCH Architecture

## Current Pipeline

Frame
→ RT-DETR Nano Detection
→ StrongSORT+ Tracking
→ Cognitive Identity Layer
→ Memory System
→ Occlusion Reasoning
→ Global Identity Manager
→ Rendering + Metrics

---

## Detection

RT-DETR Nano:
- FP16 optimized
- zero-copy preprocessing
- GPU normalization and resize
- ~21ms inference latency

---

## Tracking

StrongSORT+ includes:
- fused appearance + IoU matching
- ByteTrack-inspired low-confidence recovery
- C-BIoU buffered association
- direction-aware cost penalties
- pseudo-depth reasoning

---

## Cognitive Identity States

ACTIVE:
currently visible identity

THINKING:
temporarily lost but expected to return

FROZEN:
identity protected during overlap

PHANTOM:
invisible trajectory prediction after occlusion

RECOVERED:
successfully reattached identity

DEAD:
expired identity

---

## Memory System

Active Memory:
short-term active identity memory

Warm Memory:
long-term compressed memory for recovery

Identity Fingerprint:
- embeddings
- motion profile
- velocity history
- color histograms
- body shape

---

## Current Research Direction

SELFWATCH is transitioning from:

tracking-by-detection

toward:

trajectory-first cognitive identity ownership.
