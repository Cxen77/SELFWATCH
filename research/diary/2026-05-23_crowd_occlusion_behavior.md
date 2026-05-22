# Crowd Occlusion Behavior Observation

During testing, a tracked person entered a dense crowd region.

Observed behavior:
- identity became FROZEN
- temporary ID changes occurred multiple times during occlusion
- original identity was successfully restored after exiting the crowd

This suggests:
- cognitive persistence and ownership memory are functioning
- long-term identity continuity is partially preserved
- temporary visual continuity remains unstable during high ambiguity

The tracker currently prioritizes preserving long-term ownership over temporary visual stability.

Main issue observed:
temporary ownership oscillation during frozen-state ambiguity.