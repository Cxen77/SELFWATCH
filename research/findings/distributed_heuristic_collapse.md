# Finding: Distributed Heuristic Ownership Causes Architectural Collapse

Date: 2026-05-23

## Finding

When multiple subsystems independently manage overlapping aspects of identity
ownership, the system exhibits "heuristic collapse" — contradictory behavior
that cannot be debugged by examining any single subsystem.

## Evidence

SELFWATCH v0.7 had 6 subsystems managing ownership:
- Tracker: `is_confirmed/lost/tentative`
- Pipeline: `ACTIVE/THINKING/PHANTOM/DEAD`
- Identity manager: `_local_to_global`, proposals, provisional
- Occlusion manager: `frozen_gids`, cooldown, exit trajectories
- Arbitration layer: `_visual_owners`, shadows, stability
- Evaluator: independent per-frame metric counters

Result: internal metrics reported 100% continuity while visual output
showed 22 visible ID switches and 125 duplicate box frames.

## Root Cause

Each subsystem was correct within its own scope:
- The tracker correctly maintained local tracks
- The identity manager correctly mapped local→global
- The occlusion manager correctly detected overlaps
- The arbitration layer correctly suppressed duplicates
- The evaluator correctly measured visual output

But NO single system had a complete, canonical view of identity state.

## Generalizability

This failure pattern is likely common in any system that:
- Evolved organically through feature additions
- Added state management to each new subsystem independently
- Never established a central authority for shared state
- Reports metrics from different subsystem scopes interchangeably

## Recommended Pattern

**Central State Machine:**
- Single canonical state per entity
- All transitions logged forensically
- Other subsystems QUERY state but never SET it
- Explicit transition guards, not distributed heuristics
