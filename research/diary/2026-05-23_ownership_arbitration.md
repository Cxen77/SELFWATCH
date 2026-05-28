# Research Diary — 2026-05-23: Ownership Arbitration Architecture

## Discovery: Internal Uncertainty Leaks Into Visual Output

The SELFWATCH evaluation suite confirmed what was visually obvious:
the tracker was showing 125 duplicate-box frames and 22 visible ID switches
while internal metrics reported 0 ID switches and 100% continuity.

The root cause is now clear: the system maintained multiple ownership
hypotheses (ACTIVE tracks + THINKING tracks + provisional tracks) and
rendered ALL of them simultaneously. This is correct from a tracking
perspective — you want to preserve every possible identity. But from
a human visual perspective, it creates chaos.

## Key Insight: Separation of Concerns

The critical architectural insight is that **internal tracking** and
**visual presentation** have fundamentally different goals:

- **Internal tracking** should be GENEROUS: maintain every plausible hypothesis
- **Visual presentation** should be CONSERVATIVE: show only one winner per region

These goals are not just different — they are sometimes contradictory.
A system that aggressively persists THINKING boxes is better at long-term
identity recovery, but worse at short-term visual stability.

## Implemented: Ownership Arbitration Layer

Created `memory/ownership_arbitration.py` with three systems:

1. **Visual Dominance Resolver** — picks single winner per overlap region
2. **Shadow Hypothesis Tracker** — keeps losers alive internally (30 frames)
3. **Track-Aware NMS** — prevents duplicate births near established tracks

The arbitration layer sits between identity state machine and rendering.
It does NOT modify any tracking state — it only filters what gets drawn.

## Architecture Decision: Why Not Fix The Tracker Instead?

An alternative approach would be to make the tracker itself more aggressive
about killing duplicate tracks. I rejected this because:

1. Killing tracks permanently destroys information
2. The THINKING mechanism IS working — it successfully recovers identities
3. The problem isn't that duplicates exist, it's that we SHOW them
4. By keeping hypotheses alive but hidden, we get the best of both worlds

This is analogous to how a human brain maintains competing perceptual
hypotheses but only surfaces one conscious percept.

## Files Changed
- `memory/ownership_arbitration.py` (NEW — full arbitration system)
- `memory/occlusion_groups.py` (enhanced: frozen cooldown, exit trajectory)
- `memory/global_identity.py` (enhanced: cooldown rejection, exit region gating)
- `memory/reasoning.py` (enhanced: restricted_gids filtering)
- `engine/pipeline.py` (integrated: arbitration before render)
- `trackers/strongsort_tracker.py` (enhanced: track-aware birth NMS)

## Research Notes to Write
See recommendations at end of this entry.
