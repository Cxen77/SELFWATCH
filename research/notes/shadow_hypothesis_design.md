# Shadow Hypothesis Design Notes

Date: 2026-05-23

## Concept

A "shadow hypothesis" is an ownership claim that has been visually suppressed
but remains internally alive for potential recovery.

## Motivation

When two identities overlap (e.g., during crowd crossing), the ownership
arbitration layer picks ONE winner for visual display. The loser's box is
NOT rendered, but the loser's identity data is preserved as a shadow.

## Properties

- Shadows are invisible to the human viewer
- Shadows maintain: last box, state, velocity, age
- Shadows decay after 30 frames of suppression
- If the dominant identity fails, a shadow can instantly take over
- Shadows are logged for forensic analysis

## Why This Matters

Without shadow hypotheses, suppressing a visual duplicate would DESTROY
the identity information permanently. If the suppressed identity was actually
correct, the system would have no way to recover it.

With shadow hypotheses, the system maintains maximum information retention
while presenting minimum visual noise.

## Relationship to Cognitive Science

This is directly inspired by the concept of "preconscious processing" in
cognitive psychology. The brain maintains multiple perceptual interpretations
simultaneously, but only one reaches conscious awareness. The others remain
as "preconscious" alternatives that can rapidly surface if the dominant
interpretation fails.

## Implementation

Shadow hypotheses are tracked in `OwnershipArbitrationLayer._shadow_hypotheses`.
Each shadow stores: box, state, age, velocity, last_seen_frame.
