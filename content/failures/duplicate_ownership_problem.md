# Failure Analysis: Duplicate Ownership Persistence

Date: 2026-05-23

## Problem

The tracker frequently produced duplicate boxes for the same person.

This became the dominant failure mode after introducing stronger identity persistence mechanisms.

## Cause

The issue is caused by overly aggressive persistence of:
- THINKING tracks
- PHANTOM tracks
- frozen ownership states

while simultaneously allowing:
- new ACTIVE track creation

This creates temporary parallel ownership representations.

## Result

The tracker visually displays:
- overlapping boxes
- duplicate identities
- ghost ownership

even when internal global identity consistency is preserved.

## Observation

The system currently prioritizes:
identity preservation

over:
visual cleanliness.

## Future Direction

Potential future fixes:
- stronger duplicate suppression
- smarter PHANTOM expiration
- ownership merge logic
- adaptive persistence decay
- conflict-resolution gating