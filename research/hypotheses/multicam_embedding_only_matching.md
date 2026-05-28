# Hypothesis: Embedding-Only Cross-Camera Matching is Sufficient for Phase 1

**Date:** 2026-05-27
**Status:** Active — To be validated
**Phase:** Phase 1

---

## Hypothesis

Cosine similarity matching on OSNet embeddings alone, with temporal gating and dormant confidence decay, is sufficient for reliable cross-camera person re-identification in controlled multi-camera deployments.

## Assumptions

1. Camera viewpoints are not extremely different (e.g., not aerial vs. ground-level)
2. Person appearance does not change drastically between cameras (no wardrobe change)
3. The number of concurrent people is moderate (< 20 across all cameras)
4. Camera transition times are < 5 minutes
5. OSNet MSMT17 embeddings generalize across moderate viewpoint changes

## Expected Outcome

- **True match rate**: ≥ 70% of legitimate cross-camera transitions correctly identified
- **False match rate**: ≤ 10% of cross-camera matches are incorrect
- **ID fragmentation**: Some identities will fragment (new ID instead of reused), but the infrastructure handles this gracefully

## Validation Plan

1. Run two cameras in overlapping coverage area
2. Walk person out of Camera A, into Camera B
3. Verify same global ID is assigned
4. Repeat with multiple people
5. Measure: true match rate, false match rate, average matching latency

## Risk Factors

- Similar-looking people will cause false matches → Phase 2: add gait/color signals
- Extreme lighting differences will degrade embeddings → Phase 2: domain adaptation
- Long transition gaps will cause timeouts → Adjustable max_dormant_time
- Sequential processing reduces per-camera FPS → Phase 2: threaded processing

## Falsification Criteria

This hypothesis is REJECTED if:
- False match rate exceeds 20%
- True match rate falls below 50%
- System produces more identity fragmentation than single-camera mode
