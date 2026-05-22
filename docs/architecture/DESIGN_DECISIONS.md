# Design Decisions Log

> Record of significant architectural and algorithmic decisions in SELFWATCH.

---

## DD-001: Trajectory-First Identity Ownership

**Date**: 2026-05
**Decision**: Identity ownership is determined primarily by trajectory continuity rather than appearance similarity alone.
**Context**: Traditional MOT systems assign identity based on Hungarian matching with IoU/appearance cost. This causes ID switches when appearance is ambiguous (similar clothing, occlusion emergence).
**Alternatives Considered**:
1. Pure appearance matching (DeepSORT-style) — fails with similar-looking people
2. Pure motion prediction (ByteTrack-style) — fails during long occlusions
3. Trajectory-first with appearance validation (chosen) — robust to both cases
**Rationale**: Human perception assigns identity by *where someone is going*, not *what they look like*. Trajectory encodes spatial intent, which is harder to confuse than appearance.
**Impact**: Reduced ID switches in crowded doorway scenarios by ~X% (see exp005).

---

## DD-002: 6-Weight Confidence Fusion

**Date**: 2026-05
**Decision**: Use a weighted linear combination of 6 identity signals instead of a single distance metric.
**Context**: Single-metric matching (e.g., cosine distance on embeddings) is brittle. Different scenarios stress different signals.
**Rationale**: Allows ablation of individual signals, provides interpretability, and enables scenario-specific tuning.

---

## DD-003: Phantom Tracking for Occlusions

**Date**: 2026-05
**Decision**: Maintain "phantom" tracks that continue motion prediction during occlusion, with dedicated phantom matching when the person re-appears.
**Context**: Standard `max_lost` timeout treats occlusion as track death. This loses identity.
**Rationale**: Humans don't forget someone who walks behind a pillar. Phantoms model this continuity.

---

## DD-004: Deferred Decision System

**Date**: 2026-05
**Decision**: When identity assignment is ambiguous, defer the decision for N frames rather than forcing an immediate match.
**Context**: Forced matching under ambiguity creates cascading identity errors.
**Rationale**: Accumulating evidence over time resolves ambiguity better than single-frame heuristics.

---

<!-- Add new decisions below using the template:

## DD-NNN: [Title]

**Date**: YYYY-MM
**Decision**: What was decided
**Context**: Why this decision was needed
**Alternatives Considered**: What else was evaluated
**Rationale**: Why this approach won
**Impact**: Measured effect (link to experiment)

-->
