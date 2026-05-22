# Freeze / Stutter Observation

The tracking pipeline sometimes becomes temporarily frozen for short moments despite overall smooth FPS.

Possible causes:
- ReID spikes during crowd scenes
- GPU synchronization stalls
- UI rendering overload
- forensic logging overhead
- frame-time jitter

Average FPS alone may not accurately represent realtime smoothness.

Future improvement:
implement frame-time variance and worst-frame profiling instead of relying only on average FPS.