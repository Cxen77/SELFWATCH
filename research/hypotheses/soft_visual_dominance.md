# Hypothesis: Soft Visual Dominance Improves Stability-Recovery Tradeoff

Hard ownership locks reduce duplicate boxes but may suppress valid recovery events.

Hypothesis:
Tracking systems should maintain multiple ownership hypotheses internally while enforcing only one dominant visible identity externally.

This may:
- preserve recovery flexibility
- reduce visual flicker
- prevent duplicate rendering
- maintain long-term identity continuity

without over-constraining recovery logic.