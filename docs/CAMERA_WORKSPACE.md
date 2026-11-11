# Current portal

Use [PORTAL_CONTROLS.md](PORTAL_CONTROLS.md) and http://127.0.0.1:8766/. The following 2 November report is preserved as the earlier implementation/performance record. Later repetitions and portal fixes are in the master log.

# Camera workspace and FPS — 2 November 2026

Open http://127.0.0.1:8765/ after starting:

```powershell
.\scripts\start-studio.ps1
# To refresh an idle project-owned server after changes:
.\scripts\start-studio.ps1 -Restart
# Explicit CPU pose fallback:
.\scripts\start-studio.ps1 -Restart -PoseDevice cpu
```

This restores the source workflow of the original CustomTkinter application in
the browser. The old `app.py` remains preserved. The new workspace uses the
combined YOLO/OSNet/original cognitive engine rather than launching RF-DETR.

## Sources and controls

Upload up to three videos and select their checkboxes. Each upload is limited
to 512 MiB, receives an immutable generated filename and SHA-256 hash, and must
decode successfully. Uploads and metadata stay in `runs/studio-uploads/`; total
upload storage is capped at 5 GiB. Offsets are explicit user inputs. Container-FPS
playback estimates do not establish camera synchronization or correct VFR timing.

**Use laptop camera** selects local camera index 0; **Start tracking** opens it.
Other local indices 1–3 are selectable. The local path reuses
`multicam.camera_stream.CameraStream` from the original project, including its
bounded latest-frame capture and close lifecycle. Browser-device selection is
also available through **Find browser cameras**. Browser permission is required
for that optional path; its pending requests time out with guidance. The in-app
browser did not complete its permission request during testing, so the local
capture path was added and tested successfully. Audio is never requested.

Use videos, local cameras, or browser cameras as a session's source group.
Each session supports up to three views and has its own identity registry and
stream epochs. This is a configuration limit, not a validated capacity claim.
The initial frame limit is 600; the allowed range is 1–3000. Stop and natural
completion release camera capture. A browser session with no input expires.

Live frames replace stale pending frames; replay drops none. Live timestamps
are server receipt/dequeue estimates, not hardware capture clocks. Inputs with
a side longer than 1280 pixels are resized with the new geometry recorded.
The UI shows actual processing FPS per view and total, stage timings, dropped
live frames, memory states and an explicit record of which mode produced the run.
Preview refresh is capped at 10 polls/second and slows when the page is hidden;
FPS counts processed observations, not repeated redraws.

Completed recordings, observations.sqlite, configurations, images, timings,
identity state and annotation tools are saved in `runs/studio-<id>/`. The
workspace links the previous combined demo, identity study, detector study and
original stages 0–11. Metrics and pending gates remain visible on the plan page.

## Compute placement and actual performance

One **persistent** executor thread owns GPU inference across sessions, so model
weights and thread-local GPU warmup are reused. YOLO and OSNet run on GPU;
capture, crop preparation, original tracking/memory, SQLite, image export and
browser display run on CPU. GPU pose runs in the same serialized inference
thread. CPU pose retains the earlier bounded one-worker CPU/GPU overlap path.

The detailed path uses all YOLO boxes/masks and RTMW whole-body landmarks for
up to four highest-scoring people per frame. Fast mode keeps the same detector
at 640 pixels, skips CPU mask-polygon conversion and skips pose. It retains
OSNet, original cognitive protection and shared IDs. It does **not** promise
hand/face landmarks. Recovery rotations remain opt-in because the overhead
comparison did not justify their default GPU cost.

Measured artifacts, which must not be conflated:

| Experiment | Result | Interpretation |
|---|---|---|
| Matched CLI replay, 64 observations: CPU pose | 4.09 observations/s; 202.2 ms mean pose | Complete recorded workload after model loading |
| Same CLI replay: GPU pose | 18.25 observations/s; 16.0 ms mean pose | Same boxes/crops; GPU calls serialized |
| Warm workspace, four alternating detailed/fast/fast/detailed trials | Detailed mean 22.39; Fast mean 39.13 observations/s | Aggregate across two recorded views, one visible person/view, 64 observations/trial |
| Initial real laptop-camera smoke | 9.6 recent processing FPS over 30 observations | Short live test before the persistent-worker optimization; 44 captured frames replaced; camera released afterward |

The workspace trials are `runs/studio-fps-comparison-v2.json`, with individual
run IDs and retained recordings. Detailed trials were 23.80 and 20.97; Fast
trials 39.70 and 38.55 observations/s. Box coordinates, scores, class labels
and Person assignments were identical across those four trials. These are
exploratory same-host measurements, not a standard benchmark, three-camera
capacity claim, or evidence of crowded-store accuracy.

V1 of the workspace profile remains archived: it exposed repeated startup cost
from creating a new GPU thread for every session. Replacing that with one
persistent worker improved this workload. Do not substitute the faster CLI
number for the website's own measured throughput.

`runs/pose-device-comparison.json` verifies numerical equivalence of CPU/GPU
RTMW on 56 person observations / 7,448 joints. For 1,438 joints scoring ≥.45 in
both outputs, the maximum coordinate difference was zero; the maximum difference
over all joints was .893 pixels and score difference below .000005. Person
assignments matched. This checks runtime conversion, **not pose accuracy**.

## Isolated GPU pose runtime

The normal environment's CPU ONNX Runtime remains intact. An optional official
PyPI `onnxruntime-gpu==1.26.0` wheel is installed under `.runtime/` with its
source URL, size and SHA-256 recorded. This CUDA-12 build matches the installed
Torch CUDA 12.8 libraries. Recent default PyPI builds use CUDA 13; selecting
the newest package blindly would be inappropriate here.
[Official ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

The CUDA session uses a **2 GiB ORT arena cap**, conservative workspace settings
and TF32 disabled. Smaller 512 MiB and 1 GiB experiments failed on this model's
convolution workspace; their logs are preserved. The arena cap is not a cap on
all GPU memory: Torch, CUDA context and other applications also consume VRAM.
Auto mode enables GPU pose only after the local equivalence marker exists and
at least 3 GiB is free at startup; otherwise CPU pose is retained. `-PoseDevice
cpu` selects it explicitly. An inference error is surfaced rather than silently
claiming a successful GPU run.

```powershell
.\.venv\Scripts\python.exe -m scripts.install_pose_gpu
.\.venv\Scripts\python.exe -m scripts.validate_pose_devices --cpu runs/cpu-pose-profile-000398 --gpu runs/gpu-pose-pilot-v3-000398 --output runs/pose-device-comparison-new.json --record-validated-runtime
.\.venv\Scripts\python.exe -m scripts.profile_studio --output runs/studio-profile-new.json
```

The installer refuses to overwrite an existing isolated runtime. Reproducing
the comparison requires first producing both runs on identical source frames.

## Verification and current limits

The regression suite covers real HTTP/video upload, invalid-file cleanup,
single/persistent inference ownership, camera stop/release, stale-session frame
rejection, strict receipt ordering, FPS denominators, bounded mailboxes and
mask-conversion skipping. Real UI upload and local-camera paths were exercised.
Recordings and their annotation tools use the existing review renderer.

The perception-quality gate remains open. Cross-camera appearance links are
experimental and lack complete independent identity labels. No metric depth,
item custody, payment reconciliation or retail alert is enabled. See the
numbered [master plan and research log](PROJECT_LOG.md) and
[overhead failure analysis](OVERHEAD_DETECTOR_STUDY.md).
