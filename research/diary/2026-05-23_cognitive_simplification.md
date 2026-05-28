# Research Diary — 2026-05-23: Cognitive Simplification Rollback

## Context

After 5+ iterations of adding complexity (momentum, hysteresis, challenger
scoring, sandbox testing, visual lock), the architecture was producing
WORSE visual results than earlier versions. The complexity was interacting
in unpredictable ways and creating contradictory ownership decisions.

## Decision: Strip Back to Core Principle

Core cognitive principle: "This is person X until overwhelming evidence."

This is how human visual perception works. We don't continuously
re-evaluate identity. We commit and persist.

## What Was Cut

### ownership_arbitration.py: 466 → ~200 lines
- Dominance scoring (was: compute 6-factor score per identity per frame)
- Confidence tracking (was: grow/decay float per frame)
- Challenger model (was: count 12 consecutive frames of challenger superiority)
- Confidence margin (was: challenger must beat incumbent by 0.25)
- Sandbox (was: test rebindings for 5 frames before visual commit)
- Visual lock (was: 60-frame stability bonus of +0.30)
- Shadow hypotheses (was: maintain latent identities with velocity/age)

What remains: overlap detection + seniority resolution. That's it.

### global_identity.py: multiple mechanisms removed
- Tiered inertia (THINKING=6, PHANTOM=5, FULL=8) → single INERTIA_BASE=5
- Momentum dict (_ownership_momentum) → removed entirely
- Momentum bonus (0/2/4/6 extra frames based on duration) → removed
- Source-specific paths (if thinking... elif phantom...) → single path

What remains: committed-track rejection + uniform inertia voting.

## The New Model

Identity lifecycle in simplified architecture:

```
Track spawns → lid allocated → PROVISIONAL (5 frames)
                                   │
        Proposals CAN change ←─────┤
                                   │
                               COMMITTED (after 5 frames)
                                   │
        Proposals ALL REJECTED ←───┤ ("sticky ownership")
                                   │
                               Track dies naturally
                                   │
                               Owner archived
```

Once committed, ownership changes ONLY through:
1. Track death (local_id dies, new local_id spawns)
2. Owner removal (archived from warm memory)
3. Impossible trajectory (checked before commit)

## Results

Visible switches: 64 (up slightly from 57 with complex system)
Stability: 0.856 (comparable to 0.866 with complex system)
Code: ~250 lines FEWER across two files

## Lesson

Complexity is not free. Each mechanism interacts with every other mechanism.
5 mechanisms = 10 pairwise interactions = impossible to tune.
2 mechanisms = 1 interaction = trivially understandable.

The cognitive approach (commit and persist) is both simpler AND produces
comparable results to continuous optimization.
