# Finding: Hard Ownership Constraints Can Destroy Recovery

Date: 2026-05-23

Aggressively suppressing duplicate ownership and temporary oscillation successfully improved visual cleanliness.

However, excessive trajectory gating and recovery suppression caused severe degradation in identity recovery capability.

The system transitioned from:
- unstable recovery
to:
- failed recovery

This reveals an important architectural balance:

Too little constraint:
- duplicate ownership
- visual instability

Too much constraint:
- deadlocked recovery
- permanent identity loss

Future systems likely require:
soft uncertainty-aware dominance instead of binary ownership locking.