# Failure Analysis: Crowd ID Oscillation

Date: 2026-05-23

## Failure Description

During dense crowd overlap:
- identities entered FROZEN state
- temporary ID flipping occurred repeatedly
- original identity later recovered correctly

The tracker preserved long-term ownership but failed to maintain temporary visual stability.

## Cause

The reasoning system continued generating identity proposals near frozen identities.

This caused:
- proposal churn
- temporary ownership oscillation
- unstable visual identity assignment

Additional contributing factors:
- weak inertia threshold
- no post-freeze cooldown
- no trajectory-constrained recovery

## Impact

Humans observed:
- unstable identity behavior
- temporary ID replacement
- visible continuity failure

even though the tracker later restored the correct identity internally.

## Current Direction

New fixes focus on:
- frozen cooldown
- trajectory-constrained recovery
- stronger ownership inertia
- proposal suppression during ambiguity