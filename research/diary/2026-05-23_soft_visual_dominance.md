# Research Diary — 2026-05-23: Soft Visual Dominance

## Discovery: Over-Constrained Recovery

Ran the v0.6 pipeline with hard ownership constraints. Results:
- Duplicate boxes: 0 (perfect)
- Retrieval success: 0.5% (catastrophic)
- Visual ID switches: 0

The hard REJECT_COOLDOWN and REJECT_EXIT_REGION gates killed valid
recovery proposals at the source. The system couldn't recover
identities after crowd separation because proposals were blocked
before they could even be evaluated.

## Key Insight: Constraint Granularity

There are two fundamentally different types of constraints:

**Physics-based constraints (should be HARD):**
- Opposite direction motion
- Speed mismatch > 4x
- Currently overlapping (FROZEN)
- Spatial impossibility (too far)

**Timing-based constraints (should be SOFT):**
- Recently unfrozen (cooldown)
- Outside predicted exit region
- Ambiguous association

Physics violations are impossible events. Timing violations are merely
improbable events that become more probable with sustained evidence.

## Architecture Decision

Converted timing-based hard gates to soft inertia penalties:
- COOLDOWN: +2 extra inertia frames (not hard reject)
- EXIT_REGION: +2 extra inertia frames (not hard reject)
- These stack: a proposal during cooldown + outside exit zone = +4 frames

The penalty approach means:
- Brief noise is filtered (doesn't survive 4+ extra frames)
- Genuine recovery is delayed but NOT blocked
- System eventually accepts valid proposals with sustained evidence

## Render Hysteresis

Added 4-frame hysteresis to visual ownership switching:
- If identity A is the committed visual owner and B becomes dominant,
  B must maintain dominance for 4 consecutive frames AND beat A's
  confidence by 0.10 margin before the visual switch occurs.
- This eliminates single-frame visual flicker entirely.

## Files Changed
- `memory/reasoning.py` — removed restricted_gids hard filter
- `memory/global_identity.py` — hard gates → soft penalties
- `memory/ownership_arbitration.py` — full rewrite with hysteresis
- `docs/architecture/evolution/v0.7_soft_visual_dominance.md` — NEW
