# Testing portal — verified 3 November 2026

Use **http://127.0.0.1:8766/**. The old 8765 address redirects here; both addresses
use one model worker. Start or reopen the portal with:

```powershell
.\scripts\start-studio.ps1
# Refresh an idle service after code changes:
.\scripts\start-studio.ps1 -Restart
```

The helper refuses to restart a running or paused analysis. Stop it first so
capture closes and its observations are saved.

## Demo and recordings

The initial screen opens the existing paired retail recording without loading
models. **Play / Pause**, the timeline, previous/next observation, Reset and
playback speed all operate on recorded evidence. They remain under the camera
views. The player uses each observation's original timestamp and its matching
image; it does not interpolate people or count redraws as inference FPS.

## Uploaded footage

Choose **Videos**, upload one or more files, select up to three views and click
**Analyse selected videos**. A new upload batch becomes the selected batch;
older files remain available. A staged camera cannot conflict with Videos mode.
Time offsets and the maximum processed-frame limit are under Analysis settings.
Container FPS is only a playback estimate; zero offsets do not synchronize views.

While analysis runs, the transport provides **Pause analysis / Resume analysis**.
After saving, the same area becomes the inline recording player. The full
annotation review remains linked. Video uploads stay local and retain the
512 MiB per-file and 5 GiB upload-store limits.

## Cameras

Choose **Cameras**, then **Start laptop camera** to open local index 0 and start
analysis immediately. It no longer requires adding a source and then finding a
different Start button. The first use loads/warms models before opening capture.

**Detect** probes indices 0–3 in time-bounded hidden subprocesses and lists only
devices that returned a frame. Probes release capture; they do not run inference
or save camera images. Additional views can be added from that verified list.
This computer's verification found one available local camera, index 0.

Windows local capture reuses the original CameraStream with DirectShow and a
requested 1280×720 resolution; actual returned geometry is recorded. The old
class's default API remains compatible. Browser permission-based capture is an
optional alternative under Browser camera access, with a bounded permission wait.

**Pause analysis** stops new inference; camera capture remains open with bounded
latest-frame buffering, as the UI states. Up to the existing bounded in-flight
work can finish. **Resume analysis** continues with fresh live evidence.
**Stop analysis** releases capture and saves a replay. Model allocations remain
cached for subsequent sessions. Manual pause/resume is retained in controls.json.

## Layout, FPS and verification

Source modes are separate; advanced fields scroll inside the source panel.
The video canvas fits complete frames rather than scrolling through a cropped
frame. Playback, FPS and person-memory controls fit the tested narrow app pane.
CSS stacks the panels at smaller widths.

FPS is measured processing throughput. Recorded playback is labeled Recorded;
completed analysis is labeled Last processing FPS. Fast mode explicitly skips
pose and mask polygons, and unsupported overlay controls are disabled.
Later warm-run repetitions varied materially from the earlier peak results;
see the master log for all trials instead of treating the largest number as
capacity. No OS power/security settings were changed.

Verified through the actual browser:

- Demo Play advanced; Pause held the timestamp; seeking and stepping changed the intended observation.
- Camera detection found index 0. One-click capture showed actual frames and Person 1.
- Live Pause and Resume changed processing state; Stop closed the original capture and saved 260 observations in `runs/studio-8e5a98fd2c4b09d8/`.
- A real rank1_video.mp4 upload selected the new file without selecting the older rank0 file. `runs/studio-e2a801d1980cf36f/` completed all 32 source frames.
- Uploaded replay sought to frame 31, stepped back to 30, reset, played at 0.5× and paused at frame 15 / 2.5 seconds.

The final CPU regression suite has 101 passing tests, including redirect,
pause/resume, source probing, revision responses and development-only data
selection. Nine original multicamera checks also pass. These functional checks
do not establish independent perception/identity accuracy.

The single living checklist and experiment ledger is [PROJECT_LOG.md](PROJECT_LOG.md).
