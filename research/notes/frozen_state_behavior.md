# Frozen State Behavior Notes

The previous frozen-state system successfully preserved long-term identity ownership but still allowed temporary visual instability.

Problem:
The reasoning system continued generating proposals near frozen identities, creating proposal churn during ambiguity.

Result:
- temporary ID flipping
- unstable visual ownership
- crowd oscillation behavior

Important insight:
Protecting final ownership alone is insufficient.

The system must also suppress temporary reassignment pressure during ambiguity.

Current improvements:
- frozen cooldown
- stronger inertia
- proposal suppression
- trajectory-based exit validation

Current goal:
Maintain temporary visual continuity while preserving long-term ownership persistence.