# Finding: Visual Metrics Exposed Hidden Tracking Failures

Date: 2026-05-23

## Observation

The original SELFWATCH internal tracking metrics reported:

- ID Switches: 0
- Tracking Continuity: 100%

However, the newly developed human-perceived evaluation framework detected:

- Visible ID Switches: 22
- Teleportations: 11
- Duplicate Box Frames: 125

This revealed that the tracker was internally maintaining global identity consistency while still producing visually unstable tracking behavior.

## Important Insight

Internal tracker consistency does not guarantee human-perceived identity continuity.

The tracker frequently repaired ownership internally after temporary failures, which caused traditional MOT metrics to ignore visible instability.

Humans still observed:
- identity flickering
- ownership teleportation
- duplicate boxes
- temporary wrong identity assignment

even when the tracker later recovered internally.

## Main Failure Type

The dominant failure mode was not raw identity switching.

The primary issue became duplicate ownership persistence caused by:
- THINKING tracks remaining alive too long
- PHANTOM persistence overlap
- ACTIVE + THINKING identity coexistence

This produced high duplicate-box counts despite low internal ID-switch counts.

## Conclusion

Human-perceived evaluation is necessary for cognitive trajectory-based tracking systems.

Traditional MOT metrics alone are insufficient for evaluating visual identity continuity.