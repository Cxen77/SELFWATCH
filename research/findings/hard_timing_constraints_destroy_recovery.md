# Finding: Hard Timing Constraints Destroy Recovery Capability

Date: 2026-05-23

## Finding

Hard-blocking identity proposals based on timing conditions (cooldown periods,
exit region predictions) eliminates visual duplicate boxes but also eliminates
valid identity recovery.

## Evidence

v0.6 results after adding hard REJECT_COOLDOWN and REJECT_EXIT_REGION:
- Duplicate box frames: 125 → 0
- Retrieval success: ~2.4% → 0.5%
- Recovery capability: Functional → Broken

## Analysis

Identity recovery relies on the same proposal pathway that produces
duplicate boxes. Hard-blocking the pathway eliminates BOTH:

```
Proposals → [HARD BLOCK] → No duplicates AND no recovery
```

## Conclusion

Timing-based constraints (cooldown, exit region) should be expressed as
soft penalties (increased inertia frames) rather than hard blocks.

Only physics-based constraints (opposite direction, spatial impossibility)
should be hard gates, because they represent genuinely impossible events.

## Recommended Pattern

| Constraint Type | Expression | Example |
|---|---|---|
| Physics violation | HARD REJECT | Opposite direction, frozen overlap |
| Timing uncertainty | SOFT PENALTY | Cooldown period, exit region distance |
| Visual conflict | RENDER FILTER | Ownership arbitration, hysteresis |

This three-layer pattern separates concerns and prevents constraint
interactions from creating unintended deadlocks.
