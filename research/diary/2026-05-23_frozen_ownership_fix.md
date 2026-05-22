# Frozen Ownership Oscillation Fix

Date: 2026-05-23

During testing, a major crowd-occlusion issue was observed.

Observed behavior:
- person entered dense crowd
- identity became FROZEN
- temporary ID flipping occurred multiple times
- original identity was eventually restored correctly after crowd exit

This showed that:
- long-term ownership persistence was functioning
- temporary visual continuity remained unstable during ambiguity

Root cause investigation revealed:
- frozen identities were protected from direct rebinding
- however, the reasoning module still generated ownership proposals near frozen identities
- these proposals continuously churned through the inertia system
- temporary ownership oscillation occurred before stabilization

Additional issues:
- no post-freeze cooldown existed
- THINKING inertia was too weak (2 frames)
- no physically plausible exit-region validation existed

Implemented fixes:
- added 8-frame POST_FREEZE_COOLDOWN
- increased frozen-state inertia from 2 → 5 frames
- added frozen/cooldown proposal suppression
- added exit trajectory prediction
- added exit-region recovery validation
- added restricted GID filtering
- added new forensic rejection reasons

New rejection reasons:
- REJECT_COOLDOWN
- REJECT_TARGET_COOLDOWN
- REJECT_EXIT_REGION

Main architecture shift:
The system now prioritizes trajectory-constrained ownership recovery instead of immediate appearance-driven reassignment during crowd ambiguity.