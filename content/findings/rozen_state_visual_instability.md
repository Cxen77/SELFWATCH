# Finding: Frozen-State Protection Alone Is Insufficient

Date: 2026-05-23

The previous SELFWATCH frozen-state system successfully preserved long-term ownership continuity.

However, testing revealed that visual continuity could still become unstable during crowd ambiguity.

Important observation:
Even when final ownership remained correct, temporary proposal churn still caused visible ID flipping.

This exposed a major distinction between:
- final ownership consistency
and:
- temporary visual continuity stability

Main insight:
Protecting identities from permanent rebinding is not enough.

The system must also suppress temporary reassignment pressure during ambiguity.

This suggests that visual continuity requires:
- ownership inertia
- proposal suppression
- trajectory-constrained recovery
- ambiguity cooldown handling