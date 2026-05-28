# Finding: Separation of Internal Tracking and Visual Presentation

Date: 2026-05-23

## Key Finding

Internal tracking consistency and visual presentation stability are
fundamentally different optimization targets that require separate
processing layers.

## Evidence

SELFWATCH v0.5 evaluation results:
- Internal ID switches: 0
- Internal tracking continuity: 100%
- Visible ID switches: 22
- Duplicate box frames: 125
- Identity stability score: 90.2%

The tracker was internally perfect but visually unstable.

## Root Cause

The system rendered ALL ownership hypotheses simultaneously:
- ACTIVE tracks (real detections)
- THINKING tracks (predicted boxes during occlusion)
- Provisional tracks (newly spawned, not yet confirmed)

When two hypotheses occupied the same spatial region, both were drawn,
creating duplicate boxes and visual ownership confusion.

## Architectural Implication

Tracking systems need a dedicated **visual arbitration layer** that:
1. Receives all internal hypotheses
2. Detects spatial conflicts
3. Picks a single visual winner per region
4. Keeps losers as shadow hypotheses (internal only)

This is analogous to how the human visual system resolves perceptual
ambiguity — multiple hypotheses compete, but only one reaches consciousness.

## Generalizability

This finding likely applies to ANY multi-object tracking system that:
- Maintains multiple association hypotheses
- Uses prediction-based identity persistence (Kalman, trajectory cones)
- Renders all tracked states without filtering

The standard MOT evaluation pipeline (MOTA, IDF1) cannot detect this
problem because it evaluates internal association matrices, not visual output.
