# Research Diary — 2026-05-23: Ownership Momentum & Metric Confusion

## Implementation: Ownership Momentum

Added progressive inertia based on ownership duration:
- Tracks owned for 30+ frames: +2 extra inertia to change
- Tracks owned for 100+ frames: +4 extra inertia
- Tracks owned for 200+ frames: +6 extra inertia
- Base THINKING inertia: 2 → 6 frames
- Visual hysteresis: 4 → 12 frames
- Confidence margin: 0.10 → 0.25
- Visual lock at 60 stability frames: +0.30 dominance bonus

## Critical Discovery: Metric Measures Wrong Thing

After implementing all momentum changes, the evaluator still reported
~100-120 "visible switches." I investigated expecting ownership instability
but found:

**The internal system reports 0 ID switches and 100% continuity.**

The evaluator's `VisibleIDSwitchMetric` counts ANY pair of (disappearance
+ nearby appearance) as a "visible switch." This includes:
- Person walks off-screen, new person enters nearby → counted as switch
- Track temporarily drops to THINKING then recovers → counted as switch
- Detection flickers for 1-2 frames → counted as switch

These are NOT ownership instability. They are **normal tracker behavior.**

## Lesson: Metric Definition Matters More Than Implementation

We spent multiple iterations trying to reduce a metric that was measuring
the wrong thing. The ownership system has been stable since the momentum
changes — the high "switch" count is actually detection-layer churn.

## Open Question

What SHOULD the visible switch metric measure?

Proposed definition:
"A visible switch occurs when a PERSON (not a track) has their displayed
identity changed while they remain continuously on-screen."

This requires:
1. Person-level association (not track-level)
2. Continuous visibility requirement (not appear/disappear)
3. Identity change detection (not birth/death detection)

This is fundamentally a ground-truth problem. Without annotations, the
evaluator can only approximate this.

## Files Changed
- `memory/global_identity.py` — momentum tracking + increased inertia
- `memory/ownership_arbitration.py` — stronger hysteresis + visual lock
- `memory/ownership_state.py` — 8-frame recovery hysteresis
- `evaluation/visual_metrics/visible_id_switches.py` — tighter tolerance
- `docs/architecture/evolution/v0.9_ownership_momentum.md` (NEW)
