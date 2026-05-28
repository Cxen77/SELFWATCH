# Hypothesis: Three-Layer Constraint Architecture Optimally Balances Stability and Recovery

Date: 2026-05-23

## Background

Single-layer constraint systems produce degenerate behavior:
- No constraints: duplicate boxes, visual chaos
- All hard constraints: clean visual but broken recovery
- All soft constraints: delayed but still occasionally unstable

## Hypothesis

A three-layer constraint architecture optimally balances stability and recovery:

**Layer 1: Physics-based HARD gates (in identity reasoning)**
- Opposite-direction rejection
- Speed mismatch gating
- Active-overlap freeze
- Spatial impossibility

**Layer 2: Timing-based SOFT penalties (in identity reasoning)**
- Cooldown period → extra inertia frames
- Exit region distance → extra inertia frames
- These delay decisions without blocking them

**Layer 3: Visual RENDER filter (in arbitration layer)**
- Render hysteresis (N-frame consistency)
- Confidence margin requirement
- Single-owner dominance per spatial region

## Predictions

1. Layer 1 alone prevents physically impossible assignments (always correct)
2. Layer 2 alone delays timing-sensitive assignments (reduces noise)
3. Layer 3 alone presents clean visual output (hides remaining uncertainty)
4. Combined: clean visual + functional recovery + physics consistency

## Status: Active — testing
