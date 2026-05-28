# Multi-Camera Cross-Camera ReID — Baseline Findings

**Date:** 2026-05-27
**Category:** Findings
**Phase:** Phase 1

---

## Finding: Cosine Similarity Threshold Selection

### Context
Cross-camera person re-identification requires matching embeddings extracted from different camera viewpoints. Appearance varies across cameras due to:
- Lighting differences
- Camera angle and perspective
- Resolution and color calibration
- Partial occlusion at camera boundaries

### Threshold Analysis

| Threshold | Expected Behavior |
|-----------|-------------------|
| 0.60      | High recall, many false positives (different people matched) |
| 0.65      | Moderate recall, some false positives |
| **0.70**  | **Balanced — Phase 1 default** |
| 0.75      | Conservative, may miss legitimate cross-camera matches |
| 0.80      | Very conservative, only near-identical appearances matched |

### Rationale for 0.70

OSNet x1.0 (MSMT17) embeddings are L2-normalized 512-dim vectors. Cosine similarity of 0.70 corresponds to a cosine distance of 0.30, which:
- Accepts same-person across moderate viewpoint change
- Rejects different people with similar clothing in most cases
- Balances false positive rate against cross-camera recovery rate

### Known Risk
Similar-looking people (same clothing, similar build) may produce false matches at 0.70. This will be addressed in Phase 2 with additional signals (gait, color histograms, temporal constraints).

---

## Finding: Temporal Gating is Critical

### Problem
Without temporal gating, a person who left Camera A 30 minutes ago could match a completely different person appearing in Camera B, simply because embeddings are similar.

### Solution
Enforce `max_time_gap = 300 seconds` (5 minutes) by default.

### Evidence
In single-camera SELFWATCH, identities are held for:
- THINKING: 5-45 frames (1-9 seconds)
- PHANTOM: up to 90 frames (6 seconds)
- WARM: decaying over ~30 seconds

Cross-camera transitions typically happen within 1-120 seconds. 300 seconds provides generous headroom while preventing stale matches.

---

## Finding: Dormant Decay Rate Selection

### Design
Dormant identities decay exponentially: `confidence *= 0.998` per tick (every 30 frames).

At 15 FPS with ticks every 30 frames:
- After 1 minute: confidence ≈ 0.74
- After 3 minutes: confidence ≈ 0.40
- After 5 minutes: confidence ≈ 0.22
- After 10 minutes: confidence ≈ 0.05 (expires)

This provides a ~10 minute window for cross-camera matching, with progressively increasing uncertainty.

---

## Finding: Shared GPU Model Architecture

### Observation
Running separate detector + ReID models per camera would require:
- Camera × (detector memory + ReID memory)
- RTX 4060 (8GB): supports ~2-3 independent model instances

### Decision
Share one detector and one ReID model across all cameras. Process cameras sequentially.

### Impact
With 2 cameras: effective FPS per camera ≈ single_camera_fps / 2
With 3 cameras: effective FPS per camera ≈ single_camera_fps / 3

This is acceptable for Phase 1. Phase 2 should explore:
- Batched multi-camera inference
- Alternating ReID (skip frames per camera)
- TensorRT optimization for lower per-frame latency
