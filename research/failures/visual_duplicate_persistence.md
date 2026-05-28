# Failure Analysis: Visual Duplicate Persistence

Date: 2026-05-23

## Problem

Despite freezing, cooldown, and trajectory gating, the tracker STILL produced
duplicate boxes because the render pipeline drew ALL ownership states without
filtering for spatial conflicts.

## Root Cause

The rendering pipeline (line 534-562 in pipeline.py) collected boxes from
two sources without checking for spatial overlap between them:

1. `active` dict (ACTIVE tracks from StrongSORT)
2. `_id_states` dict (THINKING tracks from identity state machine)

When a person's appearance changed (e.g., sitting→standing), the tracker
created a NEW active track (new local_id) while the OLD thinking track
persisted. Both got added to `display` dict and both got rendered.

## Why Previous Fixes Didn't Fully Solve This

- **Frozen cooldown**: Only prevents rebinding PROPOSALS, doesn't prevent
  duplicate RENDERING
- **Trajectory gating**: Only gates recovery path, doesn't affect render path
- **Proposal suppression**: Prevents wrong identity transfer, but the
  duplicate boxes still appear from two separate identity states

## Fix

Added `OwnershipArbitrationLayer` as a pre-render filter:
- Detects overlapping entries in the display dict
- Picks single dominant winner per overlap region
- Moves losers to shadow hypotheses (internal only)

## Lesson

**Fixing the association/identity layer alone is insufficient.**
**The render layer needs its own conflict resolution.**

This is a fundamentally different kind of fix — it's not about tracking
better, it's about SHOWING tracking results better.
