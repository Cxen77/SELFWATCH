# Hypothesis: Trajectory-Constrained Recovery Improves Crowd Stability

Traditional appearance-based recovery allows identities to rebind to nearby tracks immediately after crowd overlap.

This causes:
- temporary ID oscillation
- ownership churn
- unstable visual continuity

Hypothesis:
Restricting recovery to physically plausible exit trajectories may improve temporary visual stability during crowd ambiguity.

The system now:
- caches velocity and position at freeze entry
- predicts likely exit trajectory
- restricts recovery to plausible exit regions
- adds cooldown periods after unfreezing

Expected result:
- reduced temporary ID flipping
- improved ownership inertia
- reduced crowd-induced oscillation
- more human-like continuity behavior
