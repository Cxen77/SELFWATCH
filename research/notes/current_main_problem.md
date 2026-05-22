# Current Main Problem

The current dominant failure mode is duplicate ownership persistence.

The tracker now strongly preserves identity continuity through:
- THINKING states
- PHANTOM tracking
- frozen ownership

However, this aggressive persistence sometimes allows:
- old ownership states
- new active tracks

to coexist simultaneously.

This creates:
- duplicate boxes
- ghost ownership
- overlapping identities

The system currently prioritizes identity persistence over visual cleanliness.

Future improvements may require:
- adaptive persistence decay
- smarter merge logic
- stronger duplicate suppression
- conflict-resolution gating