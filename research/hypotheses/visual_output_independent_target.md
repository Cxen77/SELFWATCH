# Hypothesis: Visual Output as Independent Optimization Target

Date: 2026-05-23

## Background

Traditional multi-object tracking systems optimize a single objective:
internal association consistency (measured by MOTA, IDF1, etc.).

SELFWATCH's evaluation suite revealed that internal consistency does NOT
guarantee human-perceived visual stability.

## Hypothesis

**Visual output quality is an independent optimization target that requires
its own dedicated processing layer, separate from internal tracking logic.**

Specifically:
- Internal tracking should maintain multiple ownership hypotheses
- These hypotheses should compete and evolve over time
- But the VISUAL output should always present a single, stable winner
- The visual layer acts as a "presentation filter" over internal uncertainty

## Predictions

1. Adding an ownership arbitration layer between tracking and rendering
   will reduce visible duplicate boxes WITHOUT changing any tracking logic.

2. Shadow hypotheses (suppressed visual identities maintained internally)
   will improve recovery quality because the system retains more information
   than a single-hypothesis tracker.

3. The gap between internal metrics (ID switches = 0) and visual metrics
   (visible switches = 22) should narrow significantly.

## Experiment Design

- Baseline: v0.5 pipeline (no arbitration layer)
- Treatment: v0.6 pipeline (with ownership arbitration)
- Metrics: Visible ID switches, duplicate box frames, teleportation events,
           identity stability score
- Control: Same video, same config, same detector

## Status: Active — implementing and testing
