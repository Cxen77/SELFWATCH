# Experiment: Frozen Ownership Stability Fix

Date: 2026-05-23

## Problem

During crowd overlap, identities entered FROZEN state correctly but temporary ID oscillation still occurred.

Observed behavior:
- identity became FROZEN
- temporary ID flipping occurred multiple times
- original identity later recovered correctly

This showed that long-term ownership persistence worked, but temporary visual continuity remained unstable.

## Root Cause

The reasoning module still generated ownership proposals near frozen identities.

These proposals continuously churned through the inertia system and created temporary ownership instability during ambiguity.

Additional issues:
- no post-freeze cooldown
- weak 2-frame inertia
- no exit trajectory validation

## Implemented Changes

Added:
- 8-frame post-freeze cooldown
- 5-frame frozen-state inertia
- proposal suppression for restricted GIDs
- exit trajectory prediction
- exit-region plausibility checks
- restricted GID filtering

## Expected Result

Expected improvements:
- reduced temporary ID oscillation
- improved crowd stability
- stronger ownership inertia
- more physically plausible recovery behavior