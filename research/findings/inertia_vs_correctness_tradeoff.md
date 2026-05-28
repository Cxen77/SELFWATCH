# Finding: Strong Ownership Inertia Can Preserve Incorrect Identity Assignment

Date: 2026-05-23

Increasing ownership inertia successfully reduced rapid identity oscillation.

However, excessive inertia also caused incorrect ownership assignments to persist much longer.

Observed behavior:
- wrong identities became visually stable
- recovery frequency decreased
- ownership became resistant to correction during ambiguity

This revealed an important tradeoff:

Higher persistence improves stability only if ownership correctness is already reliable.

Otherwise, strong inertia amplifies incorrect assignments.

Future direction:
adaptive uncertainty-aware inertia rather than globally increased ownership stiffness.