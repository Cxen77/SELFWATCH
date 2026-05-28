# Central State Machine Design Notes

Date: 2026-05-23

## State Design

Five states for identity ownership lifecycle:

```
VISIBLE_ACTIVE    [0]  — Real detection, confirmed, currently rendered
VISIBLE_FROZEN    [1]  — In occlusion group, rendered but locked
LATENT_CANDIDATE  [2]  — Track lost, within hold window, NOT rendered
RECOVERING        [3]  — Recovery accepted internally, hysteresis pending
ARCHIVED          [4]  — Moved to warm memory, no longer tracked
```

## Why These Specific States?

Previous system had ACTIVE/THINKING/PHANTOM/DEAD. The new states map to:

| Old State | New State | Key Difference |
|---|---|---|
| ACTIVE | VISIBLE_ACTIVE | Same semantics |
| ACTIVE (in frozen set) | VISIBLE_FROZEN | Previously implicit; now explicit |
| THINKING | LATENT_CANDIDATE | Explicit "not rendered" semantics |
| (none) | RECOVERING | NEW: hysteresis between latent and visible |
| PHANTOM/DEAD | ARCHIVED | Collapsed into single terminal state |

The critical addition is RECOVERING — this state allows the system to
accept a recovery proposal internally while still hiding it from the
visual output until it survives hysteresis. Previous system had no
equivalent; recovery went from THINKING → ACTIVE instantly.

## Transition Guards

All transitions have guards:
- `activate()`: Any state → VISIBLE_ACTIVE (always allowed)
- `freeze()`: Only VISIBLE_ACTIVE → VISIBLE_FROZEN
- `unfreeze()`: Only VISIBLE_FROZEN → VISIBLE_ACTIVE
- `to_latent()`: Only VISIBLE_ACTIVE/FROZEN → LATENT_CANDIDATE
- `begin_recovery()`: Only LATENT_CANDIDATE → RECOVERING
- `complete_recovery()`: Only RECOVERING → VISIBLE_ACTIVE (after N frames)
- `cancel_recovery()`: Only RECOVERING → LATENT_CANDIDATE
- `archive()`: Only non-visible → ARCHIVED

## Forensic Logging

Every transition is logged with:
- Frame number
- GID
- From state
- To state
- Reason string
- Timestamp

This makes it possible to reconstruct the complete ownership history
of any identity from start to finish.
