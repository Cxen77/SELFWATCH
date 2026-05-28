# Research Diary — 2026-05-23: Centralized Ownership Architecture

## Discovery: Distributed Heuristic Collapse

Running the v0.7 pipeline revealed a systemic architectural problem:
different subsystems disagreed about identity ownership state.

The tracker said "track 5 is confirmed active."
The occlusion manager said "gid 5 is in cooldown."
The arbitration layer said "gid 5 is a shadow hypothesis."
The evaluator said "gid 5 was visually stable."

All four statements were simultaneously true within their respective scopes,
but no single authority resolved the contradictions.

## Root Cause: Organic Growth Without Central Authority

Each feature addition (frozen state, cooldown, soft penalties, visual
arbitration, shadow hypotheses) created its own local state that partially
overlapped with other subsystems. The result was "heuristic collapse" —
too many overlapping rules producing contradictory behavior.

## Key Insight: State Machine Centralization

The fix is not to add MORE heuristics but to centralize ownership
authority into a single state machine where:
- Every identity has exactly ONE canonical state
- All transitions are explicit and logged
- Other subsystems QUERY but never SET state
- Conflicting information is resolved before reaching other systems

## Architecture Decision: Gradual Migration

Rather than replacing the legacy `_id_states` dict immediately (which
would risk breaking the entire pipeline), the SM is integrated alongside it:
- v0.8: SM runs in parallel, logs transitions independently
- v0.9: Render pipeline switches to SM as source of truth
- v1.0: Legacy `_id_states` removed

## Metric Consistency Resolution

The close() output now explicitly separates:
1. INTERNAL TRACKER METRICS (association-level consistency)
2. OWNERSHIP STATE MACHINE (canonical state summary)
3. VISUAL EVALUATION (evaluator suite - via app.py)

Each labeled with what it measures and what it does NOT measure.

## Files Changed
- `memory/ownership_state.py` (NEW — central state machine)
- `engine/pipeline.py` (SM integration, metric labeling)
- `docs/architecture/evolution/v0.8_central_ownership_sm.md` (NEW)
