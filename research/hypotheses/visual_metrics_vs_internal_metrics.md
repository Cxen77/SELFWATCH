# Hypothesis: Human-Perceived Continuity Differs From Internal Tracker Consistency

Traditional MOT metrics primarily evaluate internal association consistency.

However, a tracker may internally preserve global identity ownership while still producing visually unstable behavior such as:
- visible identity switching
- duplicate boxes
- teleportation
- temporary ownership replacement

Hypothesis:
Human-perceived identity continuity is fundamentally different from internal tracker continuity.

This suggests that traditional MOT metrics alone are insufficient for evaluating cognitive trajectory-based tracking systems.

The new SELFWATCH evaluation framework attempts to measure:
- visible continuity
- ownership stability
- physically plausible identity persistence

instead of only internal label consistency.