# SELFWATCH master plan and research log

**Updated: 14 November 2026**

**Current stage: 4**

**Status summary:** Stage 4's second frozen overlap replay cleared the pair precision and coverage targets. Gate C remains open on verified-link count, non-overlap handoffs and same-view recovery. Store commissioning is pending a floor survey and measured camera timing.

This is the living record for the project. Update its checkboxes, stage table
and experiment notes after work is verified. Mark a stage complete only when
its completion condition is satisfied. Link numerical claims to retained
artifacts. Keep failed experiments and distinguish implementation, numerical
equivalence, independent accuracy and deployment capacity.

## Purpose and research question

SELFWATCH's core job is to detect and track people across multiple cameras, keep
their shared IDs consistent, and recover tracks when people reappear. Measure
identity continuity and false links on independent labels. Body and hand pose
are optional; retail item custody remains an extension of the tracking system.
Later, combine calibrated zones and transaction evidence to produce an unresolved
exit review. Payment, intent and a future action must not be inferred as facts
from a person approaching an exit.

The proposed research question is whether uncertainty-preserving, revisable
cross-camera identity memory improves downstream custody and exit review under
fixed inference/storage budgets. This is a hypothesis to test, not established
novelty. Combining YOLO, ReID and memory alone is not a new contribution.

## Stage checklist

The numbers below follow Section 11 of the original
`Desktop/SELFWATCH_RESEARCH/EXECUTION_PLAN_V2.md`. The site derives its phase
table from this file.

| Stage | Work | Status | Evidence and completion condition |
|---|---|---|---|
| 0 | Freeze and inventory | Done | Original commit preserved; hardware, dependencies, fuzzy checkpoint and old modules inventoried. |
| 1 | Replay and contracts | Done | Timestamped camera/epoch observations, bounded ordered replay, durable idempotent store and regression checks implemented. |
| 2 | Perception pilot | Done | Gate B passed for the declared two-view retail replay. Visibility, optional landmarks and scheduling evidence are in E23. |
| 3 | Camera geometry | Done (reference rig) | Three public views and replay timing verified, with CPU floor projection, overlap topology, commissioning tools and simulated zones. E24 records error and remaining site measurements. |
| 4 | Global identity memory | In progress | E25 and E26 record the frozen overlap tests and independent handoff evidence. |
| 5 | Item and interaction evidence | Not started | Needs supported products, causal take/put/transfer evidence and annotated interaction evaluation. |
| 6 | Revisable custody | Not started | Needs event dependencies, candidate ownership, bounded retention and verified rollback/recompute after corrections. |
| 7 | Exit / POS integration | Not started | Needs directional zones, simulated transactions and late/canceled/group-payment scenarios with evidence-backed review. |
| 8 | Targeted fine-tuning | Pilot trained, not promoted | Nine epochs completed; matched evaluation and transfer check retained. Significant errors and transfer failure prevent default integration. |
| 9 | Depth / forecast extensions | Not started | Requires reliable geometry first, then causal depth/forecast baselines and measured incremental benefit. |
| 10 | Research evaluation | Not started | Needs locked settings, independent complete labels, ablations, adverse slices and confidence intervals. |
| 11 | Paper and release | Not started | Complete only when claims, novelty, limitations and reproducibility match retained experimental evidence. |

### Stage 0 — freeze and inventory

- [x] Preserve original project and Git baseline `09f6f3ed1c7668cdee99a8a8a5ee0c6b1a55fee2`.
- [x] Work on `codex/v2-replay-perception`; retain old code and experiment directories.
- [x] Record RTX 4060 Laptop GPU, 8,188 MiB VRAM, i7-13620H and 16 GB host RAM.
- [x] Locate the fuzzy project's exact SH17 YOLO checkpoint and specialized body/PPE classes.
- [x] Audit original tracking, phantom, warm-memory, ownership and multicamera modules.
- [x] Preserve dependency/model provenance and the original project context archive.

### Stage 1 — replay and contracts

- [x] Explicit boxes, optional polygons and landmark schemas in original input-pixel coordinates.
- [x] Camera ID, stream epoch, frame index, event/receipt times and timestamp method.
- [x] Merge replay by recorded event time with bounded buffering.
- [x] Accept explicit irregular source timestamps; reject inconsistent counts/order.
- [x] Preserve model-local class names instead of assuming class zero means person.
- [x] Immutable SQLite observations with duplicate/conflict checks.
- [x] Tests for ordering, reconnect scope, coordinates and persistence.

### Stage 2 — perception pilot

- [x] Integrate YOLO26n segmentation, RTMW 133-point whole body and OSNet.
- [x] Restore original identity protection through injected observations, without a second RF-DETR call.
- [x] Obtain paired RetailAction samples and verify source timestamp availability.
- [x] Build recorded review, landmark/mask toggles and independent-label export/evaluation tools.
- [x] Download an independently annotated overhead subset before model-based selection.
- [x] Lock development / validation / sealed final-test scene groups.
- [x] Compare 640, 960, four-orientation and selective-recovery perception on identical images.
- [x] Record quality, timing, failure cases and frozen development selection.
- [x] Add uploads, camera selection, laptop-camera capture and saved review to the site.
- [x] Display actual per-view/total processing FPS and pipeline stage timings.
- [x] Profile CPU/GPU pose, verify numerical equivalence and bound GPU memory allocation.
- [x] Use one persistent GPU worker and remeasure Detailed/Fast modes.
- [x] Profile crowded public frames and isolate detector, pose, ReID, memory and persistence costs.
- [x] Validate CUDA-graph ReID with changing real inputs and compare complete tracking outputs against eager execution.
- [x] Offer Tracking, Balanced and Detailed modes with explicit optional-landmark limits.
- [x] Expand the development-only labelled image pool to 280 images across the same seven development scenes.
- [x] Decode/export and inspect expanded person targets across small-person and crowded overhead layouts.
- [x] Score independent MOT17 visibility flags and visually reviewed retail boxes, including clipped people.
- [x] Evaluate a separate domain-trained person detector against its own pretrained baseline; retain gains and regressions.
- [x] Download a frozen public whole-body validation subset with independent body/hand labels and preserve source/licensing metadata.
- [x] Measure GT-crop versus detector-conditioned body/hand PCK, with missing detections retained in denominators.
- [x] Compare bounded hand refinement on CPU and CUDA; save raw/gated accuracy, score coverage and incremental latency.
- [x] Measure body and hand landmark accuracy by size and reviewed viewpoint, using COCO visibility for body and wrist visibility as the hand occlusion proxy.
- [x] Calibrate the hand model on development data and validate its bounded schedule with held-out cached predictions and combined retail replay.
- [x] Evaluate crowded and multi-person footage with independent MOT17 labels and qualify the declared retail replay views.
- [x] Freeze the qualified accuracy and latency operating point in `docs/phase2-final-protocol.json` and pass Gate B for that scope.

### Stage 3 — camera geometry

- [x] Verify published WILDTRACK calibration and run a frozen, offline floor-plus-appearance linking experiment on a separate time segment.
- [x] Pin the three public camera views, native dimensions, published time basis and replay validity window.
- [x] Load verified published intrinsics/extrinsics and implement surveyed floor fitting with optional pinhole distortion correction.
- [x] Define and test aisle, shelf, checkout and directional exit zones on the reference map.
- [x] Project visible foot contacts with uncertainty. Withhold clipped, out-of-footprint, stale and uncalibrated observations.
- [x] Validate the surveyed-target and measured-clock code with separate synthetic fit/check data. Retain the real-frame footpoint and cross-view diagnostic.
- [ ] **Store commissioning** — record installed camera models and view layout, survey independent physical check targets, capture shared sync events, and verify the actual aisle, shelf, checkout and exit geometry. Required before enabling store geometry decisions.

### Stage 4 — global identity memory

- [x] One outward-facing global Person-ID authority; old engine IDs are camera-local tokens.
- [x] Restore StrongSORT, phantom/warm retrieval, fingerprints, ownership locks and shadow audit.
- [x] Keep clean per-view galleries; reject low-confidence and overlapping appearance crops.
- [x] Add overlapping-view appearance candidates, ambiguity margins and abstention.
- [x] Preserve held owners without rendering predicted people as observed.
- [x] Compare fixed perception inputs across baseline, original and weak-birth policies.
- [x] Add an online IDF1 evaluator that rejects incomplete ground truth and future-alias leakage.
- [x] Score a bounded, independently labelled three-camera WILDTRACK pilot with a camera-local control.
- [x] Compare geometry-only and geometry-plus-appearance controls while preserving online assignments and recording sampled ID changes.
- [x] Implement a CPU replay policy using calibrated time/coverage, floor uncertainty, appearance and prospective link revocations.
- [ ] Obtain complete independently reviewed identities across crowded synchronized views.
- [x] Measure false cross-view pair matches and online IDF1 on a temporal holdout.
- [x] Apply Stage 3 geometry and time constraints in the fixed-input replay.
- [x] Score labelled handoff and return opportunities on a separate public camera pair and measure the local-tracker bottleneck.
- [x] Compare immediate, confirmed, interval and revisable association with fixed perception.
- [x] Measure local owner preservation and a dormant handoff candidate on fresh public sequences.
- [x] Meet overlap pair precision and coverage targets on the second frozen three-camera test.
- [ ] Verify adequate cross-camera handoff coverage and recovery delay on independently labelled footage.
- [ ] Pass Gate C with sufficient verified links and coverage on a fresh holdout.

### Stages 5–7 — interaction, custody and POS

- [ ] Stage 5: define supported SKUs and unknown-product behavior; acquire suitable labels.
- [ ] Stage 5: implement causal take/put/transfer evidence and cross-view event deduplication.
- [ ] Stage 5: evaluate interaction precision/recall on visible and occluded cases.
- [ ] Stage 6: create an event ledger with source observations and dependency indexes.
- [ ] Stage 6: maintain bounded alternative custody states and reversible corrections.
- [ ] Stage 6: test identity correction, returns and handoffs without duplicate events; pass Gate D.
- [ ] Stage 7: create a POS simulator and transaction schema with completion, cancellation and reversal.
- [ ] Stage 7: test late events, outages, partial/group baskets and directional exits.
- [ ] Stage 7: evaluate an evidence-backed unresolved-exit review queue; pass Gate E.

### Stage 8 — targeted fine-tuning

- [x] Prepare a person-box pilot export using development/validation scenes only.
- [x] Preserve final-test scenes and define a bounded 4060 training recipe.
- [x] Expand the development-only person-box pilot to 280 images and audit its targets.
- [ ] Acquire representative supported-camera training and evaluation labels before the main experiment.
- [x] Train one small person detector separately; record checkpoint/license/seed/configuration.
- [x] Compare to the same architecture's pretrained model with identical inputs, settings and timing scope.
- [ ] Tune confidence/calibration on validation; freeze before final testing.
- [ ] Repeat important learned comparisons across seeds when feasible.

This early pilot responds to Stage 2's measured detector errors. It does not
mean Stages 3–7 are complete or that a trained checkpoint has passed deployment gates.

### Stages 9–11 — extensions, evaluation and paper

- [ ] Stage 9: establish constant-velocity/Kalman forecasting and calibrated floor-map baselines.
- [ ] Stage 9: compare low-rate causal depth and compact probabilistic forecasts.
- [ ] Stage 9: retain extensions only if downstream benefit justifies total compute cost.
- [ ] Stage 10: lock datasets/settings and evaluate independent held-out outcomes.
- [ ] Stage 10: run oracle identity/interaction, memory-policy and budget-matched ablations.
- [ ] Stage 10: report adverse slices, dropped frames, frame age, false reviews and uncertainty intervals.
- [ ] Stage 11: audit closest prior art; narrow or reject the novelty claim if needed; pass Gate F.
- [ ] Stage 11: write methods, results, limitations and reproducibility from the experiment ledger.
- [ ] Stage 11: prepare an appropriately scoped release and demonstration.

## Decision gates

| Code | Gate | Status | Completion condition |
|---|---|---|---|
| A | Visible evidence | Open | Target footage actually resolves the hands/items required for the supported task. |
| B | Perception | Passed | Person detection in two fixed overhead retail replay clips on the RTX 4060 laptop. 91.7% recall, 100% precision on 24 sampled boxes and 7.3–8.0 FPS/view. Optional landmarks remain diagnostic. |
| C | Identity | Not passed | Independently measured handoff coverage, false merges and recovery reliability. |
| D | Custody | Not evaluated | Better ownership results than an immediate top-1 baseline at matched budgets. |
| E | Exit review | Not evaluated | Truthful payment evidence and enough normal activity to measure false reviews. |
| F | Research contribution | Not evaluated | Close prior-art review and controlled experiments substantiate the claimed contribution. |

## Working implementation and assumptions

- Repository: `C:/Users/Dev/Desktop/SELFWATCH`; research library: `C:/Users/Dev/Desktop/SELFWATCH_RESEARCH`.
- Python 3.11 environment and exact packages: `requirements-v2-lock.txt`. Model paths/hashes appear in every run's `run.json`.
- Models: `weights/v2/yolo26n-seg.pt`, `weights/v2/rtmw/wholebody133.onnx`, `weights/osnet/osnet_x1_0_msmt17.pth`.
- YOLO segmentation and RTMW landmarks are separate capabilities; pose points are not anatomical pixel masks. Hidden fingers are not invented observations.
- ReID uses appearance, not names or a face-recognition database. Similar clothing can require no-match/ambiguity.
- The original camera-local cognitive engine receives YOLO/OSNet evidence; it cannot trigger hidden detector/ReID inference. SharedIdentity alone assigns global Person IDs.
- Appearance admission: detection score ≥.4, no strong person overlap, usable crop size; scheduled fresh extraction. Shared candidates use .78 similarity, .06 margin and repeated support. These thresholds are experimental.
- Original low-score birth restrictions remain in the main path. Experimental repeated weak births are a separate policy comparison, not proven identity improvement.
- CPU owns capture, preprocessing, tracking/memory, storage and display. One persistent thread serializes GPU inference. CPU pose uses a bounded worker; GPU pose stays in the GPU owner thread.
- Detailed studio mode estimates up to four people per frame. Fast mode keeps boxes, ReID and identity memory while skipping pose and mask-polygon conversion.
- Live capture replaces stale pending frames; replay drops none. Live times are receipt/dequeue estimates, not measured hardware synchronization.
- The app is local at `http://127.0.0.1:8766/` (8765 redirects here); start with `.\scripts\start-studio.ps1`. Source uploads/camera actions are explicit; natural completion and Stop release capture.
- There is no validated metric depth, product custody, POS settlement or operational retail alert yet.

## Experiment and decision ledger

The entries below consolidate existing artifacts and work verified on 2 November
2026. Old plan/review dates remain in their source files; no unrecorded start/end
times are asserted here.

### E01 — source audit and literature

The fuzzy model was a specialized SH17 YOLO checkpoint, so its body/PPE classes
cannot be attributed to every pretrained YOLO model. RF-DETR's configured box
interface did not supply full hand anatomy. Preserve the old source and compare
matching capabilities rather than choosing a detector by name alone.

The research library now contains **32 verified original PDFs** with source URL,
byte size, page count and SHA-256 in `paper_manifest.json`, indexed in
`PAPERS_INDEX.md`. Literature notes are in `RESEARCH_PLAN.md`. New local dataset
references are `papers/WEPDTOF_WACV2022.pdf` (10 pages) and
`papers/LOAF_ICCV2023.pdf` (11 pages). These older foundations supplement the
recent-paper review; their dates must remain explicit.

### E02 — restored combined cognitive pipeline

Artifact: `runs/combined-cognitive-v2-000398/`; details:
`docs/OLD_PROJECT_INTEGRATION.md` and `docs/REID_AND_SCHEDULING.md`.

Recorded two-view replay combines YOLO, RTMW, OSNet and original memory.
The retained 64-observation run has 54 tracked person observations, 56 posed
person observations and one experimental cross-view link. Both final views
display Person 1. This is an integration demonstration with one shopper,
not ground-truth identity accuracy. Topology and contradictions remain advisory;
gait is not used on sparse irregular samples.

### E03 — fixed-input identity policy study

Artifact: `runs/identity-study-pilot-v3/`; protocol: `docs/IDENTITY_STUDY.md`.

Fourteen paired clips provide 848 camera observations and 577 person detections.
The same cached perception/appearance evidence is fed to a motion baseline,
original cognitive engine and experimental repeated-weak-birth policy.
Assigned detections are 577 / 496 / 533; distinct local tokens are 58 / 24 / 27.
These are coverage and fragmentation diagnostics, **not accuracy**: suppressing
IDs can reduce both token count and assignment coverage. The maximum detected
occupancy is three people per view. Independent complete identity labels are
still required before reporting IDF1 or claiming fewer genuine ID switches.

### E04 — independently annotated overhead detector comparison

Dataset: `datasets/wepdtof-pilot/`; artifact: `runs/overhead-detector-pilot-v1/`;
protocol/details: `docs/OVERHEAD_DETECTOR_STUDY.md`.

Before inference, select eight uniform-midpoint images per source clip: 128
images from 16 clips / 14 scene groups. Keep segments of one video together.
Use 56 development images / seven groups; 32 validation images / three groups;
40 sealed final-test images / four groups. Final-test files are checksum-verified
but not decoded, inferred, exported to training or shown in the comparison.
Range requests read 197,435,892 bytes of the 5,435,073,011-byte official archive.
All saved files have checksums; dataset notices are preserved.

Protocol: confidence .25; person-center ROI filtering; identical neutral-114 ROI
masking for every model; rotated source annotations clipped and converted to
enclosing axis-aligned boxes; one-to-one matching at IoU .5. This is a custom
pilot, **not the authors' rotated-AP benchmark**. There are 262 validation targets.

| Candidate | TP / FP / FN | Precision / recall | F1 | Mean detector ms |
|---|---|---|---|---|
| Native 640 | 61 / 21 / 201 | .744 / .233 | .355 | 60.6 |
| Native 960 | 92 / 31 / 170 | .748 / .351 | .478 | 78.2 |
| Four orientations 640 | 83 / 48 / 179 | .634 / .317 | .422 | 230.1 |
| Weak-view rotation 640 | 61 / 17 / 201 | .782 / .233 | .359 | 140.5 |

Development selected four orientations before validation. Its validation
generalization was poorer than 960. Keep that negative result; do not silently
replace the frozen choice. No candidate met the 50 ms development P95 budget.
V1 uses the lower empirical P95 quantile; its exact source is archived. New
runners use nearest-rank P95 and automatic source snapshots.

Decision: rotation remains off by default; prioritize domain adaptation and
small-person error analysis. Tracker changes cannot recover never-observed
people. Higher validation recall is still far below a deployment-quality result.

### E05 — memory-triggered detection recovery

Artifact: `runs/combined-memory-recovery-016699/`.

Recent missing-owner regions can request a same-model cardinal-rotation probe.
Requests are camera/epoch/frame scoped, expire after two seconds and are throttled
at .5 seconds per camera. Memory does not create detections; actual model outputs
are inverse-mapped to the original coordinates and score-preserving NMS is used.

The 32-observation integration run made 16 probes / 80 detector passes, with nine
missing-memory-region requests. It produced no cross-view link in that short
sample. This verifies execution, not a tracking or accuracy improvement.

### E06 — training preparation, not a trained model

Artifact: `datasets/wepdtof-detector-pilot/`.

Export 56 development images / 698 person boxes and 32 validation images / 262
boxes. Exclude sealed final-test scenes. Preserve ROI preprocessing, source/pixel/
label hashes, `data.yaml`, notices and a bounded recipe: batch 4, 640 pixels,
20 epochs, seed 42, zero loader workers. This is a pipeline pilot; more suitable
training data is required for the main comparison. Person boxes cannot supervise
segmentation masks, hand landmarks, products, metric depth or ReID. No training
or fine-tuned checkpoint exists yet.

### E07 — pose-device failures and numerical validation

Failed artifacts: `runs/gpu-pose-pilot-000398.log` and
`runs/gpu-pose-pilot-v2-000398.log`. ORT arena caps of 512 MiB and 1 GiB failed on
a convolution-workspace request of 686,824,704 bytes amid other allocations.
The 2 GiB arena configuration succeeded; keep these failures in the record.

Matched successful runs: `runs/cpu-pose-profile-000398/` and
`runs/gpu-pose-pilot-v3-000398/`; comparison:
`runs/pose-device-comparison.json`.

| Runtime | Observations | Wall seconds after model loading | Offline observations/s | Mean pose ms |
|---|---|---|---|---|
| CPU pose, bounded CPU/GPU overlap | 64 | 15.650 | 4.09 | 202.20 |
| GPU pose, serialized GPU calls | 64 | 3.506 | 18.25 | 15.99 |

Both use the same person crops and produce 56 posed person observations.
Across 7,448 joints, 1,438 score ≥.45 in both outputs; confident-joint maximum
coordinate difference is zero. Maximum over all joints is .893 pixels, maximum
score difference below .000005, and Person assignments agree on every frame.
This is numerical equivalence, **not independent pose accuracy**.

Runtime provenance: isolated official PyPI `onnxruntime-gpu==1.26.0`, CUDA 12
compatible with Torch CUDA 12.8; wheel size 226,539,455 bytes and SHA-256
`cc5329aad02d9745cc3ae9cdb185bfa1aad242a7bf89b8c471280002ec40f98a`.
The normal CPU environment remains intact. The 2 GiB arena cap excludes Torch,
CUDA context and other applications; it is not total GPU memory consumption.
Auto mode requires the local validation marker and at least 3 GiB free at startup.

### E08 — website worker and FPS profile

Artifacts: `runs/studio-fps-comparison.json` (initial implementation) and
`runs/studio-fps-comparison-v2.json` (persistent GPU worker); the files retain
individual run IDs and stage timings.

The first implementation recreated an inference thread per session and incurred
repeated GPU startup cost. On its warmed-weight trials, Detailed averaged 5.51
and Fast 7.61 observations/s. Retain this failed implementation's measurements.

After persistent-worker reuse, run a warmup then Detailed / Fast / Fast / Detailed
on the same 64 timestamped paired observations per trial, one visible person per
view. Detailed trials: 23.80 and 20.97 observations/s, mean **22.39**. Fast trials:
39.70 and 38.55, mean **39.13**. These are aggregate rates across two recorded views.
Raw boxes, scores, class labels and Person assignments agree across all four
trials. Fast intentionally omits pose and mask polygons. The UI additionally
reports recent processing FPS per view and total; preview redraws are not counted.

This is an exploratory same-host replay profile, not crowded-store or three-camera
capacity. Unrelated OS activity, short trials and only two repeats limit inference.
Preserve the distinction between standalone CLI, website, live capture, recent
rate and whole-run throughput when writing results.

### E09 — real source controls and capture lifecycle

The original CameraStream is reused for local capture. Native and browser-camera
selection, upload progress, source offsets, maximum frame limits, Detailed/Fast
options, Stop and saved review are implemented at `http://127.0.0.1:8765/`.
The optional browser camera request did not complete in the in-app browser; a
bounded timeout and local-capture path address that behavior.

Real laptop-camera smoke: `runs/studio-32d6eb309b60ee53/`, 30 processed frames,
9.6 recent processing FPS and 44 stale captured frames replaced. This was before
the persistent-worker improvement. The capture closed after completion. It is
functionality evidence, excluded from the public dataset experiment.

Real UI upload: a local `rank0_video.mp4` was selected through the file picker,
uploaded, decoded and persisted. `runs/studio-5efa49df9e968cd3/` completed an
eight-observation source-processing smoke. Its container-FPS playback estimate
is explicitly distinguished from the dataset sample's supplied irregular times.
Detailed UI/control documentation: `docs/CAMERA_WORKSPACE.md`.

### E10 — checks and reproducibility checkpoint

- [x] CPU/GPU numerical-equivalence experiment passed on matched observations.
- [x] Actual file-picker upload and processing completed.
- [x] Actual local-camera capture completed and released the device.
- [x] Warm website Detailed/Fast profile completed with identical box/ID assignments.
- [x] Final v2 regression suite: 94 tests passed, including master-log/site synchronization.
- [x] Nine original multicamera checks passed again; retained log is `runs/original-multicam-check.log`.
- [x] `pip check`, JavaScript syntax checks and `git diff --check` passed after the implementation update.
- [x] Run final regression/checks after the master-log/site synchronization change; site returns all 12 stages and six gates from this file.
- [x] Save a source checkpoint and manifest for later paper reproduction: `runs/research-checkpoint-20261102/`.

Current supporting artifacts remain in their original directories. Source
checkpoint: `runs/research-checkpoint-20261102/`, containing changed source/docs,
checksums and original Git baseline, without datasets, model binaries or private
camera images. Git work is not represented as a published release.

## Continuation log — 3 November 2026

### E11 — later FPS replication and bounded development expansion

The later warmed website repetitions were slower than E08: `runs/studio-fps-comparison-v3.json`
recorded Detailed mean 8.39 and Fast mean 12.84 observations/s. V4, after smaller
live packets and shared revision-encoded responses, recorded Detailed mean 9.63
(trials 8.11 / 11.15) and Fast mean 12.13 (12.15 / 12.11). Software/client state
and host conditions differed; do not treat these as a controlled causal ablation
or replace the earlier results selectively. E08's 22.39 / 39.13 values are historical
warmed-workload observations, not stable capacity guarantees. GPU clock/load
samples are saved in `runs/fps-hardware-samples.json`; they do not establish a
single cause for the variation. The UI must continue to show actual processing
FPS, independently of redraws and recorded playback.

Lazy detector, pose and full ReID API warmups now occur in the loading phase.
Synthetic warmup is never tracked, stored or displayed as camera evidence.
`runs/studio-e905d8d0e37b1b42/` retained the first-session 64-observation result
(9.110 seconds after loading); it must not be described as the faster warm case.

`datasets/wepdtof-development-expanded/` is complete: 280 labelled development
images, selected uniformly and unioned with the original development samples,
across only the same seven locked development scenes. Download payload: 322,216,436 bytes. Validation and sealed-test
images were not added. The extension records its parent selection hash, source
ETag, member CRC/SHA-256 and bounded transfer manifest. Preparation and training
of an expanded export remain pending. No fine-tuned weights exist.

### E12 — portal failure report and corrective work

The user reported that camera options did not work, the layout was poor and Play /
Pause were missing. Inspection found the familiar port 8766 refusing connections;
the earlier source screen ran on 8765, used a separate recorded-review screen,
and allowed confusing mixed video/camera selection. These are usability failures
that are not negated by earlier isolated smoke checks.

Corrective work implemented and browser-verified: one canonical portal on 8766 with 8765 redirecting;
separate Videos / Cameras / Demo source choices; immediate Start laptop camera;
bounded detection of available camera indices; original CameraStream capture via
DirectShow on Windows; visible recorded Play/Pause, seek, step, reset and speed
controls; real analysis pause/resume and Stop; full-frame scaling without inner
cropping; and source controls that stay visible in the narrow app pane.

- [x] Canonical service and redirect implemented; backend redirect/pause/detection checks passed.
- [x] Recorded demo Play advanced the timestamp; Pause held it at 2.8 seconds in browser verification.
- [x] New layout inspected in the actual narrow pane; clipped video scrolling corrected.
- [x] Verify camera detection and one-click laptop capture through the new portal: one available camera (index 0), actual frames and Person 1 observed.
- [x] Verify live pause/resume/Stop releases capture and saves replay: `runs/studio-8e5a98fd2c4b09d8/`, 260 observations, controls logged and CameraStream closed.
- [x] Verify new upload selection, replay, seek/step, reset and 0.5× speed: `runs/studio-e2a801d1980cf36f/`, 32 observations. Uploaded replay paused at frame 15 / 2.5 s; narrow app-pane layout inspected.
- [x] Final implementation checks: 101 v2 tests, nine original multicamera checks, JavaScript syntax, pip check and git diff whitespace check passed.
- [x] Save a verified source checkpoint after this corrective work: `runs/research-checkpoint-20261103/` (105 changed source/docs files, model/evidence hashes; training still in progress at capture).

### E13 — expanded targets and first domain adaptation pilot

The development-only extension was exported with the original frozen parent
supplying validation: `datasets/wepdtof-detector-expanded-v1/`. Counts are 280
development images / 3,481 person boxes and 32 validation images / 262 boxes.
The exporter rejects a mismatched parent selection, development extensions
containing non-development scenes, absent validation and duplicate image IDs.
Five new partition tests passed; the complete v2 suite now passes 106 tests
(`runs/v2-check-20261103.log`).

`runs/person-target-audit-v1/audit.json` records verification of every exported
pixel hash, label hash, count and normalized box boundary. Two deterministic
development-only target sheets (smallest target and largest person count per
scene) were visually inspected. They show fish-eye distortion, people at many
orientations, dense layouts and very small targets. There are 2,509 development
boxes below 1% of full-image area, including 358 below 0.1%. These size statistics
do not measure visibility or detector quality. Occlusion labels are still absent.
No sealed-test image was decoded, exported, reviewed or inferred.

A separate official YOLO26n detection checkpoint was downloaded from the
[Ultralytics v8.4.0 asset release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt),
5,544,453 bytes, SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Size and hash match the official release API digest. The existing segmentation,
whole-body and ReID weights remain available. The
[official training API](https://docs.ultralytics.com/modes/train/) was checked
against the locally pinned package; no package upgrade was performed.

Experiment `runs/overhead-training/person-domain-expanded-v1/` uses the frozen
20-epoch maximum recipe, batch 4, 640 pixels, seed 42, zero data-loader workers,
rotation augmentation, mixed precision and patience 5. The actual automatically
selected optimizer is AdamW, initial learning rate 0.002. Its experiment lock
records source/export/audit/initial-weight hashes, versions and hardware.
An additional official checkpoint copy was fetched by Ultralytics' AMP check;
it is a software precision check, not another training dataset or trained model.
One GPU training job runs at a time; the portal is idle with cameras closed.
The hidden follow-up `scripts.finish_person_pilot` is queued to wait for successful
training, then benchmark the pretrained and best adapted checkpoints sequentially.
Its state is `runs/person-domain-comparison-v2/status.json`; the resulting
`comparison.json` / `REPORT.md` will reject mismatched split locks, protocol,
image IDs, processed pixel hashes or ground-truth targets. It stops on a training
failure and never promotes a model. Two additional comparison-integrity tests
passed. Validation is used to select the best training checkpoint; it is not an
untouched final-test result. The retained software checkpoint captures the code
and this in-progress experiment state; later runtime results remain separate.
The first waiting follow-up was canceled before any benchmark to correct a
legacy report label that assumed segmentation weights. The replacement uses the
actual checkpoint task/name; detector inference settings were not changed.
Correction after result review: `runs/v2-check-20261103-final.log` ran 108 tests
in 9.367 seconds with ONE FAILURE, rather than the previously reported full pass.
The progress-page test expected the obsolete literal `Prepared only` for Stage 8.
The subsequent 112-test run also retained that failure in
`runs/v2-check-person-results.log`. Both failed logs are preserved. The test has
now been changed to verify evolving status values using a controlled fixture;
the actual master-file check validates its complete schema without freezing a
past stage description. Compile and whitespace checks passed separately.

- [x] Expanded export and complete pixel/target integrity audit.
- [x] Inspect development target sheets; preserve figures and size counts.
- [x] Verify the separate pretrained detector's official size/SHA-256.
- [x] Launch the fixed single-seed training pilot; retain configuration and logs.
- [x] Finish training and retain best/last checkpoint hashes and early-stop logs.
- [x] Compare against its own pretrained detector using the frozen person-box protocol.
- [x] Document measured gains/failures and decide to retain the current portal model; see E14.

### E14 — completed detector pilot, failure analysis and integration decision

Training completed successfully in `runs/overhead-training/person-domain-expanded-v1/`.
Patience 5 stopped training after nine epochs; epoch 4 was best by validation
fitness. Epoch-loop time was 472.628 seconds; experiment lifecycle including
preflight/loading/final evaluation was longer. Best checkpoint SHA-256:
`2a3a23d5b10a0e431670f9772f27f4e653b1463675222e7b1591adb66dc646f2`;
last checkpoint SHA-256:
`a54fb823c2a919b13d76e54634e9fa8a338a56c18fa9e86bcda682ea0253c969`.
Both files' sizes and hashes were verified. The retained research copy,
`weights/v2/person-domain-expanded-v1/`, includes `detector.pt`, a model card,
actual training arguments and the dataset notice. Publisher licensing options
are recorded with the [official source](https://www.ultralytics.com/license);
no license agreement or account form was submitted.

The sequential comparison finished in `runs/person-domain-comparison-v2/`.
Both runs have unchanged source snapshots and identical split locks, processed
pixel hashes and ground truth. Fixed diagnostic: confidence .25, IoU .5,
ROI-neutral input, enclosing AABB targets; detector-only latency. There are
32 validation images / 262 person targets in three held-out scene groups.
This validation set selected the training checkpoint and is not untouched test data.

| Validation case | Pretrained F1 | Adapted F1 | Pretrained recall | Adapted recall | Adapted mean / P95 ms |
|---|---:|---:|---:|---:|---:|
| Native 640 | .3006 | .4848 | .1870 | .3969 | 43.2 / 59.9 |
| Native 960 | .4858 | .3077 | .3588 | .2595 | 49.4 / 60.6 |
| Four orientations 640 | .4011 | .5009 | .2824 | .5534 | 187.3 / 224.8 |
| Selective recovery 640 | .3006 | .4883 | .1870 | .4771 | 143.1 / 206.1 |

At native 640, TP increased 49→104, FP increased 15→63 and FN decreased
213→158. Paired target analysis found 74 rescued targets but 19 newly lost
targets. The development-selected quality candidate was native 960 for BOTH
checkpoints; its validation degradation must be reported, rather than choosing
640 retrospectively and calling it the winning system. No candidate met the
predeclared 50 ms detector P95 budget on development. Timing excludes pose,
ReID, camera capture and saving; it is not full-system multi-camera FPS.

`runs/person-domain-errors-v1/` contains checked per-scene and target-size counts
and a visually inspected failure figure. Native-640 adapted recall increased in
all three validation scenes, but remained .371 (exhibition), .365 (jewelry store)
and .472 (repair store). Both models recalled **0/12** targets smaller than 0.1%
of image area. For 0.1–1% targets, recall was 7/111→23/111; for larger targets,
42/139→81/139. These bins describe target area, not measured occlusion or hands.

A separate ordinary-retail transfer check used the SAME 64 timestamped
observations of paired clip 000398, unchanged GPU whole-body pose, OSNet and
original cognitive tracker. Saved report: `runs/person-transfer-report-v1/`.
The adapted model returned 57 raw person detections, all scores below .35;
none reached the existing .5 strong-birth or .4 appearance-admission thresholds,
so it produced zero assigned track observations. The existing segmentation path
returned 56 person detections and 54 assigned track observations, plus 18
non-person detections; it recorded one experimental cross-camera link.
These are functional outputs on unlabeled footage, NOT detection/ReID accuracy.
The failure shows why a detector-only F1 gain cannot justify replacing the
complete pipeline or globally weakening its identity protections.

**Decision:** do not promote the fisheye-trained detector to the portal or lower
identity thresholds to force IDs. Keep the existing general segmentation,
whole-body, OSNet and old memory integration. Preserve the trained branch for
controlled domain-specific experiments. Stage 2 and Gates A–C remain open.
Next detector work needs supported-camera labels, a mixed-domain control and
explicit confidence/admission calibration on development data; hand/body
visibility evaluation remains a separate requirement. Repeating training without
these checks would not resolve the observed transfer failure.

Verification after fixing the stale status test: **113 tests passed in 8.916 s**,
`runs/v2-check-person-results-fixed.log`; compile and whitespace checks passed.
The earlier 108/112-test failed logs are retained and explicitly corrected above.
No sealed-test image was decoded, exported, reviewed or inferred in this work.

- [x] Complete and verify training checkpoints and the matched comparison.
- [x] Inspect failure figures and quantify scene/size-specific errors.
- [x] Run ordinary-retail transfer through whole-body, ReID and original memory.
- [x] Record a decision to retain the current portal model and preserve the candidate model card.
- [x] Fix the stale progress test and verify 113 passing tests, compile and whitespace checks.
- [x] Save and verify a post-results source/evidence checkpoint: `runs/research-checkpoint-20261103-results/` (108 source/docs files, model hashes, results and preserved failed/passing test logs).

## Continuation log — 4 November 2026

### E15 — whole-body labels downloaded while evaluation tools were built

The user requested background data downloads alongside useful implementation.
The author-linked [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody)
validation annotations and a frozen 64-image diagnostic are saved in
`datasets/coco-wholebody-pilot-v2/`. Sampling used uniform midpoint strata over
sorted image IDs with at least one noncrowd body-labelled person, before any
inference; no model-based image filtering was used. All annotations for the
selected images are retained, including crowd records for explicit exclusions.

There are 255 person annotations, 250 noncrowd people and 110 valid annotated
hand parts. The annotation file is 132,970,317 bytes, SHA-256
`f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e`.
The 64 JPEGs total 10,206,692 bytes. Selection SHA-256:
`08a1523634951da40aaabc1d475f8959c2ad2d032b4d44e92ce7beb95471f757`.
Every downloaded file is hash-verified before evaluation. Original per-image
licences/URLs and the authors' research/noncommercial annotation notice are retained.
This public validation diagnostic is NOT training data or a new untouched retail test.

Transfer failures are preserved: the first request returned the large-file
download notice; the first binary transfer hit the 96 MiB cap. The source size
was then checked (132,970,317 bytes), the cap set to 160 MiB, and the incomplete
96 MiB file retained in `datasets/coco-wholebody-pilot/`. The image alias had a
TLS hostname-validation failure; the same public COCO S3 bucket was accessed
through its verified HTTPS endpoint. Completed annotations were reused while
image downloading resumed. Successful-file byte counts exclude failed-attempt
traffic; they must not be presented as the total network transfer.

- [x] Download and freeze validation annotations/images independently of model results.
- [x] Retain file/selection checksums, source notices and failed transfer records.
- [x] Implement annotation layout/validity parsing and bounded model-free tests.
- [x] Download original COCO-WholeBody (2007.11858, 28 pages) and RTMPose
  (2303.07399, 11 pages) papers, verify PDFs and retain extracted text/hashes.
  These are older supporting references, not newly published methods. The
  verified research library now contains 34 PDFs; its prior manifest is backed up.

### E16 — separate detection, body-pose and hand-localization errors

`runs/wholebody-validation-v2/` is the current frozen diagnostic. It uses the
existing YOLO26n-seg person boxes at confidence .25, RTMW 192×256, IoU .5 person
matching, and top-six detected-person pose admission. The GT-crop case poses all
noncrowd labelled person boxes for diagnosis. Wrapper calls group at most three
ROIs, but inspection of RTMLib confirms **one model forward per ROI, actual batch 1**.
The earlier v1 metadata misleadingly called the grouping `pose_batch_size=3`;
v2 corrects this wording and repeats the run. Reported body/hand PCK counts
agree between v1/v2. Both source snapshots remain unchanged during their runs.

Metrics are custom PCK diagnostics, NOT official OKS AP. Body/foot tolerance is
5% of sqrt(original person-box area); hand/face tolerance is 10% of sqrt(original
part-box area). Body-visible (v=2), body-occluded (v=1) and body-labelled (v>0)
are separate. Valid hand/foot/face parts use v>0 reliability, including fractional
values as documented by the authors; fractional hand reliability is not relabelled
as visible/occluded. Invalid/unlabelled parts supply no invented ground truth.
Missing person associations and matches excluded by the six-person crop limit
keep their eligible-joint denominators and contribute zero correct predictions.

| Diagnostic | Visible-body PCK (1,805 joints) | Hand PCK (2,301 joints) | Hand score coverage at .45 |
|---|---:|---:|---:|
| RTMW with GT person boxes | .8249 | .3751 | .9757 |
| Detector + RTMW | .7191 | .3577 | .9379 |

Of 250 noncrowd person boxes, 191 matched a detected person and 59 did not;
30 matches had no pose under the admission limit. Five crowd annotations were
excluded. The 43 unmatched predictions are not official detector false positives:
some can overlap crowds or otherwise unscored people. Hand accuracy remains poor
even with known person boxes; a better person detector alone will not solve it.
GT crops diagnose the localization dependency but are not guaranteed to give
higher pose PCK for every individual hand than a differently framed detected crop.

RTMW's hand score coverage .9757 alongside PCK .3751 demonstrates that its raw
landmark scores are not calibrated correctness probabilities. Gated PCK at .45
is reported alongside raw PCK and coverage; no confidence threshold was fitted here.

### E17 — specialized hand refinement and placement on the laptop

The official [RTMLib hand model list](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/hand.py)
provided two bounded downloads under `weights/v2/hand/`: RTMPose-m hand pose
(55,080,248-byte ONNX, SHA-256
`39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29`)
and RTMDet-nano hand detection (4,010,667-byte ONNX). Archive CRCs and local SHA-256
provenance are retained. Only the pose model was evaluated here; the separate
hand detector was downloaded but is not evaluated or enabled.

The hand experiment consumes the SAME cached detector/whole-body outputs.
Its practical ROIs use at least eight whole-body hand points scoring >=.45,
their enclosing box with minimum 16-pixel sides, clipped to the image. Up to six
hands per image are selected by median qualifying point score with deterministic
ties. No GT constructs or selects these ROIs. Selected hands replace all 21
points; unselected hands retain baseline points, with no GT-based or cross-model
score cherry-picking. A separate GT-hand-box case is diagnostic only.

| Hand case, same 2,301 reliable joints | Raw PCK | Gated PCK at .45 | Score coverage |
|---|---:|---:|---:|
| Whole-body, GT person boxes | .3751 | .3742 | .9757 |
| Whole-body detector pipeline | .3577 | .3577 | .9379 |
| Hand model, GT hand boxes | .5954 | .0500 | .0582 |
| Predicted-ROI hand refinement | .4637 | .0508 | .1169 |

CPU-only inference used CPUExecutionProvider with two intra-op threads and
sequential batch-one crops (`runs/hand-refinement-validation-v1/`). It refined
212 hand ROIs, deferred 27 under the six-hand limit, and used 110 GT hands in
the oracle diagnostic. Incremental practical refinement averaged 382.2 ms/image,
P95 720.0 ms. This is too costly for an every-frame CPU extension on this workload.

The CUDA repetition (`runs/hand-refinement-validation-gpu-v1/`) used the isolated
ORT 1.26.0 runtime, a 1 GiB hand-session arena limit and the same model/inputs/ROIs.
Incremental mean was 55.5 ms/image, P95 108.6 ms. Raw hand PCK remained .4637;
58,520 full-body/hand coordinate triples were compared across saved outputs,
with max coordinate difference .210 pixels and raw-score difference 2.921e-6.
This is numerical/provider verification and indicative separate-run timing, not
proof of accuracy from equivalence, simultaneous residency, or three-camera FPS.
The baseline detector/whole-body compute and ReID are excluded from these costs.
RTMW's 2 GiB and hand-model's 1 GiB session limits are not total process VRAM.

**Decision:** preserve this hand-refinement candidate offline. Its raw localization
improvement is useful, but reusing RTMW's .45 display threshold would hide most
specialized-model output. Do not rescale scores or select a threshold on this
already-inspected validation subset. Calibrate on development data, then evaluate
on supported retail views and validate low-rate/event-driven scheduling and
combined VRAM/latency before enabling it. Pixel decoding, crop geometry, memory,
storage and UI belong on CPU; these measurements favor GPU for the added neural
inference when it is scheduled. No production hand model or identity threshold
was changed. Stage 2 and Gates A–C remain open.

Figures were exported as PNG/SVG in `runs/wholebody-figures-v1/` and visually
inspected. Hand examples select the three largest valid GT hand areas among
matched/posed people, one per image, without selecting for improvement. Their
GT-centred zooms are review-only and are not inference crops. Missing or incorrect
predictions remain scored in full-image coordinates. The original downloaded
dataset and all measured failures are preserved for paper writing.

- [x] Run and retain whole-body oracle/detector diagnostics and corrected batch metadata.
- [x] Evaluate bounded hand-only refinement on CPU and CUDA with fixed ROIs.
- [x] Reconcile cached baseline joint counts exactly and compare providers numerically.
- [x] Export and visually inspect quantitative/qualitative research figures.
- [x] Verify 123 passing tests (`runs/v2-check-hand-refinement.log`, 9.670 s), compile and whitespace checks.
- [x] Save and verify `runs/research-checkpoint-20261104/`: 116 source/docs files, model hashes, dataset manifest, test logs, numeric results and research figures.
- [ ] Validate hand-landmark accuracy on independently labelled retail footage before enabling the optional specialist in live operation.

## Continuation log — 6 November 2026

### E18 — tracking priority, frontend run and supplementary hand experiment

The user clarified that SELFWATCH is a multi-camera video person tracking
system and asked to run the model and frontend. The portal ran at
http://127.0.0.1:8768/ on the local machine. Re-analysing the built-in two-view
retail clip processed 64 observations and recorded one cross-camera link.
The saved review at `runs/studio-a45768d723e0974f/` shows Person 1 on both
views at 9.30 s; `tracking-preview.png` is from that public recording.
These are working system outputs on an unlabelled clip. A separate live camera
session started from the frontend and was allowed to finish independently.

The in-progress hand experiment was completed as supplementary work. Two
image-disjoint 48-image subsets were frozen from unused COCO-WholeBody validation
images before inference. The old diagnostic image IDs and exact file hashes
were excluded. Selection required at least one valid labelled hand. Development
alone selected the specialist score threshold of .35, using a 75% correctness
floor and 10% coverage floor among matched, refined, labelled joints. The
policy was written before evaluation. At the frozen threshold, 331 of 426
emitted labelled joints were correct (77.7%); correct yield across all 2,488
eligible joints was 13.3%. The .45 control yielded 101 correct joints (4.1% of
all eligible). These are localization diagnostics on hand-enriched public
images. They do not measure false hands on unlabelled people or retail transfer.
See `runs/hand-calibration-policy-v1.json` and
`runs/hand-calibration-evaluation-v1.json`. No hand model was enabled in the
frontend. Scheduling an additional hand stage is deferred in favor of tracking.

### E19 — first labelled three-camera person tracking diagnostic

Downloaded a frozen subset of [EPFL WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/):
C1/C3/C5 at 18 consecutive annotated timestamps (1700–1785, step 5),
54 images and 18 source label files. The pinned ZIP required handling a 4 GiB
offset ambiguity. The bounded member reader requires the exact local filename
and validates the central CRC and uncompressed length. The transfer was
201,467,559 bytes; file hashes and source revision are in
`datasets/wildtrack-pilot-v1/manifest.json`. The original CVPR paper, extracted
text and checksum are saved in the research library (35 PDFs).

The study uses the same YOLO26n-seg boxes and eligible OSNet features for
all tracking policies. It evaluates online IDs at IoU .5 using published
projected boxes (802 boxes, 22 subject IDs). The detector matched 541 boxes.
The full-view IDF1 diagnostic charges unmatched predictions even outside the
labelled ground region; it is not WILDTRACK's official metric. In the
synchronized pair columns, coverage is reported against 686 labelled
same-person view pairs. False sharing counts differently labelled people
receiving the same ID in two matched observations at the same timestamp.

| Policy | Diagnostic IDF1 | Same person matched in both views | Correct shared IDs | False shared pairs |
|---|---:|---:|---:|---:|
| Motion/IoU + shared identity | 33.4% | 307 | 1 | 3 |
| Legacy cognitive + shared identity | 39.7% | 200 | 1 | 0 |
| Tentative weak births + shared identity | 37.1% | 229 | 1 | 0 |
| Same legacy tracks, camera-local IDs | 39.4% | 200 | 0 | 0 |

The legacy controller's near-equality with the camera-local control is a
failure of cross-camera coverage, despite the frontend replay working.
Among 686 labelled same-person view pairs, 307 were detected in both views,
226 had eligible embeddings in both, and only 2 of those reached the existing
.78 cosine threshold. Both of those cleared the .06 margin in the separate
single-timestamp comparison. This isolates an appearance bottleneck on these
camera angles; it does not justify reducing the threshold without testing
false links on a separate segment. Tracking improvements should compare
view-aware appearance evidence and camera geometry against camera-local and
appearance-only controls. No policy threshold or live default changed.

Results, frozen predictions/features, converted labels and the appearance
funnel are under `runs/wildtrack-tracking-v1/`. This is one outdoor segment
at 2 Hz; continuous handoff, store transfer, dense-view quality and live
three-camera throughput remain open. 127 v2 tests passed; the new tests cover
archive corruption, source-box conversion, pair coverage and score abstention.

## Paper-writing notes

### E20 — frozen geometry and appearance experiment, completed 6 November; audited 8 November

The calibration/evaluation subset download completed: C1/C3/C5, 12 annotated
timestamps 1300-1355 step 5, plus pinned original calibration files. Development
used the earlier 1700-1785 segment. Verified undistorted intrinsics, Rodrigues
extrinsics, centimetre-to-metre conversion and the official floor grid. Matching
uses mutual nearest floor points, ambiguity margin, two adjacent sightings and
clean appearance corroboration. Settings were selected on development before
evaluation: radius .75 m, margin .25 m, cosine .55. Development yielded 66 correct
shared pairs and 1 false shared pair; evaluation yielded 51 correct and 0 false,
against 108 same-person pairs matched in both views and 494 possible pairs.
Diagnostic IDF1 increased from .33825 to .41298. The geometry-only arm failed
the development precision gate and remains a diagnostic control.

Temporal audit is essential: sampled global-ID changes increased from 10 to 15
as provisional camera-local IDs were joined; local-ID changes remained 10.
These are sampled changes, not full-sequence CLEAR IDSW. Thus this experiment
supports better cross-view sharing, not reduced temporal switching. No final
aliases were used to improve online scores. Source consistency with published
projected boxes is not independent physical calibration accuracy. The prototype
is offline, unbounded for long runs, and not installed as a live camera policy.
Results: `runs/wildtrack-geometry-policy-v2.json`,
`wildtrack-geometry-evaluation-v2.json` and
`wildtrack-geometry-temporal-diagnostics-v2.json`.

### E21 — crowded FPS and further tracking literature, 8 November

Read and downloaded six additional primary PDFs: OA-SORT (2026), TrackTrack
(2025), ARC-SORT (October 2026 preprint), on-the-fly homography calibration
(September 2026 preprint), BoostTrack++ (2025 revision), and the older Deep
OC-SORT baseline (2023). The research manifest now has 41 verified papers.
Official TrackTrack and BoostTrack implementations were inspected and their
source revisions/checksums retained. Decisions, limitations and ordered next
experiments are in `docs/TRACKING_STABILITY_AND_FPS.md`.

The public crowded profile uses 12 WILDTRACK C1 observations, 1280x720,
8-17 detected people, predecoded images and original 2 Hz timestamps. Includes
tracking, SQLite, JPEG encoding and JSON, excludes browser/capture and final
review export. The isolated eager control measured 3.56 FPS detailed and 7.19
FPS tracking-only. Detailed costs: pose 109 ms, ReID 68 ms, detector 55 ms,
cognitive memory/association 9 ms. Initial unhandled ReID batch shapes caused
large cold stalls; cold and warm measurements are retained separately.

Added fixed warmed batch buckets and CUDA graph replay for OSNet, with a
single GPU owner and persistent buffers. Changed real-input checks at batches
1/2/4/8/16 yielded maximum embedding error 0; padding error was at most 3e-8.
Graph-enabled complete trials: detailed 4.30/4.15 FPS; balanced 6.74/6.08;
tracking-only 10.42/10.06. All six graph trials and both eager controls produced
identical local/global assignment signatures on these frames. This is numerical
and behavioral agreement, not independently established identity accuracy.
See `runs/crowded-tracking-graph-20261108.json`,
`crowded-tracking-eager-control-20261108.json` and
`reid-graph-validation-20261108.json`. The first scheduled benchmark overlapped
CPU tests and is retained as a diagnostic, not the isolated headline comparison.

Tracking is now the frontend default. ReID, original phantom/warm recovery and
shared memory remain active. Balanced rotates up to two current detections into
pose per view every .5 source seconds, with no stale keypoint reuse; Detailed
retains up to four poses per frame and mask outlines. Detection/tracking still
process every replay observation. CPU performs association/memory, decode and
delivery; GPU performs neural inference. CPU pose remains a bounded-worker
alternative. `--no-reid-graph` preserves an explicit eager comparison path.
Persistence and publication timings were added to server telemetry. Crowded
retail accuracy, longer clips and 1/2/3-camera live capacity are still open.

Verification: 139 v2 tests passed in 10.165 s; compile, JavaScript syntax and
whitespace checks passed. Geometry unit cases cover ambiguous neighbours,
two-sighting confirmation, appearance corroboration, gaps, camera collisions
and transitive-distance rejection. Scheduling tests cover view/epoch isolation,
bounded state, rotating selection and absence of stale keypoints. All 41 paper
PDF sizes/checksums match the library manifest; six new extracted texts are
nonempty. `runs/tracking-performance-20261108/` retains a regenerable figure
and source-checksummed measurement summary. Public sample frontend smoke
verification is recorded separately from labelled accuracy experiments.

Frontend smoke: Balanced run `studio-aca7697912adc9a3` and Tracking run
`studio-bb009542bd2727ac` each completed 64 observations with one cross-camera
link. Browser checks verified the mode help, fresh-landmark capability limits,
play/pause labels and final Person 1 on both public camera views. The final
Tracking run reported recent aggregate 13.39 observations/s (6.76/6.80 per view),
on the easier paired sample, not the crowded stress input. The service remains
on http://127.0.0.1:8768/. No private camera was opened. Screenshot and state
receipt: `runs/tracking-performance-20261108/portal-preview.png` and
`runs/tracking-portal-smoke-20261108.json`.

- [x] Save and verify `runs/research-checkpoint-20261108-tracking-v2/`: changed source/docs, model hashes, research manifest, frozen identity results, performance trials, test log and public UI proof. Source cap explicitly raised to 8 MiB to retain two existing ~4 MiB project-context documents; private camera recordings remain excluded.

Keep these sections ready to populate from the ledger: problem and hypotheses;
closest prior art; perception/identity/event contracts; memory admission and
revision rules; scheduling and resource budgets; datasets and annotation protocol;
causal evaluation; results and ablations; failure analysis; limitations and release.

Required comparisons still include standard tracker/ReID, original memory,
immediate top-1 versus uncertain associations, clean versus contaminated galleries,
bounded retention, geometry/no-geometry, oracle identity/interaction, and camera-only
versus simulated/real POS. Report useful coverage alongside error. Reduced ID
counts or zero alerts on a few clips do not establish success.

Independent IDF1 results now exist for the bounded WILDTRACK diagnostic (E19).
Complete labelled retail identities and a separately frozen evaluation remain open.
HOTA is not implemented in the current evaluator. Take/put accuracy, custody error,
payment reconciliation, false reviews per camera-hour, metric geometry error and
forecast error remain unmeasured. Do not fill these with detector counts or
numerical-equivalence results. Interactive camera smoke recordings are excluded from public dataset results. Keep retrospective corrections separate from the
decision that was possible online.

### E22 — continuous identity tracking on two MOT17 sequences

Downloaded 1,125 native JPEGs and the official labels from two complete
[MOT17](https://motchallenge.net/data/MOT17/) training sequences. MOT17-02
provided 600 development frames and MOT17-09 provided 525 evaluation frames.
Both have 30 Hz source timestamps. The bounded download verifies the exact ZIP
member name, length and CRC before recording SHA-256 in the dataset manifest.
The tracker processes 1280 × 720 inputs. The evaluator receives boxes mapped
back to native 1920 × 1080 coordinates.

All four policies received identical detections and eligible cached OSNet
features. The original tracker served as the baseline. One alternative required
fresh, clean features for local gallery updates. Another required a 0.03 margin
between association costs. The fourth combined both. Development selected the
margin policy after 52 → 44 switches, with HOTA 19.73% → 20.78% and recall
15.80% → 15.33%. The policy and its source hashes were frozen before scoring
the evaluation sequence. Standard metrics use [TrackEval](https://github.com/JonathonLuiten/TrackEval)
at `12c8791b303e0a0b50f753af204249e622d0281a`, with scalar dtype aliases updated
for NumPy. The original MOT17 distractor rules apply.

| Held-out MOT17-09 policy | ID switches | IDF1 | HOTA | Person recall |
|---|---:|---:|---:|---:|
| Original memory tracker | 122 | 55.01% | 40.85% | 52.90% |
| Fresh local gallery updates | 95 | 51.74% | 39.92% | 54.55% |
| Frozen 0.03 association margin | 96 | 57.79% | 42.88% | 54.22% |
| Fresh updates and margin | 111 | 50.10% | 38.91% | 54.10% |
| ByteTrack on the same detections | 36 | 57.18% | 46.31% | 62.59% |

The margin reduced identity false positives from 713 to 643 and improved
same-ID recovery from 7 to 9 of 13 labelled low-visibility episodes. The recovery
diagnostic requires two consecutive clean matches to the pre-occlusion ID within
one second of the first clean return. All four CLEAR traces agreed exactly with
TrackEval's ID-switch counts. Gallery filtering alone increased switches from
52 to 73 on development and reduced IDF1 by 3.28 percentage points on evaluation.
ByteTrack's stronger HOTA and recall support testing its motion association
alongside the existing recovery and identity memory. These results describe
single-camera tracking. Live multi-camera retail validation remains open.

A second replay ran the original and frozen margin policies with the portal's
scheduled ReID, pose disabled, and the same 525 evaluation frames. Switches
fell from 88 to 79, IDF1 rose from 39.69% to 40.32%, and HOTA rose from 33.16%
to 33.82%. Offline throughput was 13.48 and 15.88 observations per second,
respectively. Each policy ran once in a fixed order. Use those rates as replay
measurements. Repeat in alternating order to estimate a policy speed effect.
The benchmark includes decoding, tracking, SQLite, JPEG encoding and JSON
serialization. The scheduled configuration generated 366 and 374 fresh
embeddings. Its lower identity scores make ReID scheduling another candidate
for controlled tests. The source policy remains available through the offline
benchmark while supported retail footage and multi-camera validation are pending.

Saved evidence under `runs/mot17-continuity-development-v1/`,
`runs/mot17-continuity-evaluation-v1/` and `runs/mot17-bytetrack-evaluation-v1/`.
Scheduled ReID measurements are in `runs/mot17-continuity-scheduled-v1/`.
The frozen selection is `runs/mot17-continuity-policy-v1.json`. The research
branch is `codex/v2-replay-perception` on GitHub. Datasets, model files, local
recordings and raw run artifacts stay in the local workspace.

From the repository root, use the existing v2 environment and local detector
and OSNet weights. Prepare the pinned evaluator and download the two sequences
with the commands below. Use fresh output directories for each rerun.

```powershell
git clone https://github.com/JonathonLuiten/TrackEval.git .runtime/TrackEval
git -C .runtime/TrackEval checkout --detach 12c8791b303e0a0b50f753af204249e622d0281a
python -m scripts.prepare_trackeval
python -m scripts.download_mot17_native
python -m selfwatch_v2.mot_continuity cache --sequence MOT17-02-FRCNN --output runs/mot17-dev-cache
python -m selfwatch_v2.mot_continuity replay --sequence MOT17-02-FRCNN --cache runs/mot17-dev-cache --output runs/mot17-dev --cases baseline clean_memory strict_margin clean_margin
python -m selfwatch_v2.mot_policy --study runs/mot17-dev --cache runs/mot17-dev-cache --output runs/mot17-policy.json
python -m selfwatch_v2.mot_continuity cache --sequence MOT17-09-FRCNN --output runs/mot17-eval-cache
python -m selfwatch_v2.mot_continuity replay --sequence MOT17-09-FRCNN --cache runs/mot17-eval-cache --output runs/mot17-eval --cases baseline clean_memory strict_margin clean_margin --policy runs/mot17-policy.json
python -m selfwatch_v2.mot_failure_analysis --study runs/mot17-eval
python -m scripts.profile_mot17_continuity --policy runs/mot17-policy.json --output runs/mot17-scheduled
```

**Next work** compares motion association with the existing memory on labelled
retail footage, then measures longer runs on one, two and three cameras.
The camera geometry experiment improved cross-view sharing. The continuous MOT17
test measures a separate problem, maintaining one ID over time within each view.


### E23 — Phase 2 qualification, 11 November 2026

The supported operating point is person-box perception for two fixed, elevated
retail replay views of a recognizable shopper at 600 × 600 source resolution.
The existing cognitive tracker and scheduled OSNet ReID run alongside the
selected YOLO26n detector. The qualification checks person boxes at IoU 0.5.
The floors were frozen at 80% recall, 85% precision, two thirds recall in
each view and 5 processed observations per second per view. The sample has six
fixed frame indices per view per clip. All visible recognizable people were
boxed from the decoded images before model output was reviewed. This is a
single visual review of sampled frames from a public training split.

| Retail check | Configuration | TP / FP / FN | Recall | Precision |
|---|---|---:|---:|---:|
| 000398 and 000996 | Segmentation checkpoint at 640, native | 18 / 1 / 6 | 75.0% | 94.7% |
| 000398 and 000996 | Same checkpoint, fixed orientation | 23 / 7 / 1 | 95.8% | 76.7% |
| 001078 and 001844 | Same checkpoint, fixed orientation | 19 / 4 / 5 | 79.2% | 82.6% |
| Fresh 004738 and 007981 | Frozen YOLO26n detector at 640, FP16, score 0.35 and fixed orientation | 22 / 0 / 2 | 91.7% | 100.0% |

The first two checks failed the accuracy floors. Both became development input
for the next detector comparison. The development grid compared five
checkpoint and resolution combinations with five confidence cutoffs. It
selected the 640 FP16 detection checkpoint at 0.35 with 43 TP, 7 FP and 5 FN
on 48 reviewed frames. Camera rotations for the two fresh clips were assigned
from the source images before inference. `docs/phase2-final-protocol.json`
freezes the detector hash, rotations, qualification thresholds and labels.
The frozen oriented arm passed every floor on the fresh clips. The native
control with the same detector also reached 22/24. The four oriented view
recalls were 5/6, 6/6, 6/6 and 5/6. Replay achieved 7.31 and 8.00 FPS per view
for the two clips, including detection, ReID, tracking, decoding, SQLite
writes and JPEG encoding. Per-view rates divide each view's frame count by
the shared replay wall time. Wider camera layouts, crowds, capture and browser
latency require further measurement.

Independent MOT17 visibility explains the remaining detection risk. In
MOT17-09, the prior detector matched 94.5% of high-visibility targets and 22.5%
of low-visibility targets. MOT17-02 reached 58.0% and 3.0% in those same groups.
These are detector recall slices using the official visible fraction at cutoffs
0.3 and 0.7. All target misses remain in each denominator.

The disjoint COCO WholeBody evaluation set measured visible-body PCK at 66.2%
and occluded-body PCK at 25.8%. Visible-body PCK was 37.3% below a 64-pixel
person short side and 79.5% at 128 pixels or greater. Hand PCK was 27.3% below
a 16-pixel hand short side and 57.0% at 32 pixels or greater. Hand PCK grouped
by labelled wrist visibility was 41.6% with a visible wrist and 12.7% with an
occluded wrist. Wrist status is a proxy for hand occlusion. On the fixed
24-image viewpoint review, roughly level and elevated views scored 59.1% and
68.2% visible-body PCK and 34.4% and 43.2% hand PCK. The groups differ in subject
size and composition. Report the small group counts with these diagnostics.

The frozen hand cutoff of 0.35 retained 46 correct joints among 52 emitted by
the one-ROI schedule simulation, out of 2,488 labelled eligible joints. The
combined retail replay ran 19 fresh specialist calls and accepted five joint
predictions at 5.58–6.66 FPS per view. It called at most one hand ROI per camera
per half second and used only current pose observations. This records the
schedule and latency. Live specialist integration requires independently
labelled retail hands and a useful joint yield. The Phase 2 operating point
therefore uses the person detection and tracking mode.

Evidence is in `runs/phase2-retail-validation-v1/`,
`runs/phase2-retail-qualification-v1/`, `runs/phase2-retail-final-v1/` and
`runs/phase2-closeout-v1/`. Reproduce the frozen qualification from the repository
root with the v2 GPU environment and existing RetailAction inputs.

```powershell
python -m scripts.validate_phase2 --protocol docs/phase2-final-protocol.json --tracking-only --output runs/phase2-retail-final-repeat
python -m scripts.phase2_visibility
```

Stage 3 starts with actual-camera floor mapping, clock offsets and validation
against known floor points. Stage 4 will use that evidence for multi-view
identity linking and measure continuity on independently labelled views.

### E24 — Phase 3 camera geometry, 11 November 2026

**Scope.** Qualified the offline reference path on WILDTRACK views C1, C3 and
C5. Verified 36 native 1920 × 1080 frames, in 12 bundles at 2 Hz, and verified
the supplied camera calibration hashes. The intersection of the labelled floor
region and each camera's projected image defines its nominal footprint. All
three pairs have geometric overlap. Occluders can reduce actual shared
visibility. The authors report about 50 ms synchronization. The replay uses
published frame index divided by 10, a conservative assumed 50 ms uncertainty
per view, and a 150 ms pair gate. The original per-camera timing residuals
were unavailable. [Dataset and calibration](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/), [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html).

**Visible-foot diagnostic.** Marked 20 selected contacts in frame 1300 at
1280 × 720, then scaled to native pixels before scoring. Three marks were
withheld by the footprint check. Published person IDs were displayed to
establish correspondence. The published world labels were themselves
derived with the source camera calibration. This is a small, single-frame
consistency check without an independent physical ground survey. Reported
reference contact radii express sensitivity to a six-pixel input perturbation.

| Comparison | Contacts or pairs | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| C1 to published ground position | 6 | 0.24 m | 0.34 m | 0.36 m |
| C3 to published ground position | 5 | 0.26 m | 0.62 m | 0.67 m |
| C5 to published ground position | 6 | 0.17 m | 0.76 m | 0.94 m |
| Same subject in overlapping cameras | 7 | 0.22 m | 0.75 m | 0.91 m |

The reported residuals include every accepted projection. Phase 4 needs
separate evidence before using small distance thresholds or merging a global
ID on geometry. The existing offline linking results remain in E20.

**Implementation.** `selfwatch_v2/camera_geometry.py` runs on CPU without
additional inference. A calibrated camera and stream epoch produce a metric
foot contact, a pixel-sensitivity plus held-out residual radius, and aligned
capture time with an uncertainty interval. Pair coverage requires compatible
epochs, shared floor coverage and bounded timing disagreement. Zone membership
requires the uncertainty disk to clear its boundary. The directional gate
requires two confirmed observations from one track, with clearance on each
side, plausible speed and elapsed time, and passage through the finite
segment. The reference layout is simulated on the plaza map. Its ten checks
cover checkout membership, boundary abstention, direction, missing contacts,
clipping, missing cameras, a reset epoch and stale timing. The production
tracker and alert path are unchanged pending Phase 4 linking validation.

**Reproduce the reference and commission a measured rig.**

```powershell
python -m scripts.validate_phase3 reference datasets/wildtrack-geometry-evaluation-v1 docs/phase3-wildtrack-footpoints.json runs/phase3-reference-v1/results.json
python -m unittest tests_v2.test_camera_geometry
python -m scripts.validate_phase3 survey store-survey.json runs/store-survey/results.json
```

The frozen report is in `docs/phase3-reference-results.json`. The survey
input has schema version 1, a revision-specific `floor_id`, a `cameras` list,
and a `layout` with `zones` and `exit_gates`. Each camera records
`camera_id`, `stream_epoch`, `model`, `view`, native `size`, and a `lens`
with `input` set to `undistorted` or `raw`. Raw input requires a measured
`pinhole` matrix and `distortion` coefficients. Its `survey` pairs
`fit_px` with `fit_m` and `check_px` with `check_m`, with a declared
`max_error_m`. Use at least four fit targets and three different measured
check targets distributed within the same flat footprint. Its `clock` holds
`fit_events` and `check_events` as [camera seconds, rig seconds] pairs,
`quantization_s`, and `max_error_s`. Capture three or more fit events and two
separate check events within the fit time window. Each zone has a name, kind
and `boundary_m`. Each exit has a name, `start_m`, `end_m`, `hysteresis_m`,
`max_gap_s` and `max_speed_m_s`. A fixture in
`tests_v2/test_camera_geometry.py` shows the full structure.

The survey validator emits held-out spatial and timing residuals, the fitted
offset/drift and per-camera coverage. Recommission after a changed camera
position, crop or stream epoch. Record the actual site measurements and
chosen tolerances before any store geometry alert is enabled.

The synthetic survey passed the command-line check. All 156 v2 regression
tests passed. Phase 4 can use the reference coverage and timing contract
while collecting independent evidence for global ID continuity.

### E25 — reversible global identity replay, 14 November 2026

The CPU replay uses fixed local owners from the original cognitive tracker and
clean OSNet crops. It scores overlapping views only when the calibrated
footprints and capture-time intervals support comparison. Uncertainty enters
the association cost through a lower and upper distance. A missing appearance
crop remains a plausible competing match. A link needs repeated mutual evidence
and cannot give two local owners in the same camera one global ID. Sustained
incompatible geometry or expiration can revoke a link prospectively. Earlier
outputs remain as recorded. This is a replay implementation for overlapping
views. The existing shared memory provides the dormant appearance comparator.

WILDTRACK frames 1700–1785 supplied development. The first development-only
geometry-margin variant linked 5 of 200 both-matched pairs. A second variant
incorporated appearance into the candidate ranking and retained its best-case
cost for missing crops. It linked 69 of 200 pairs correctly with 3 incorrect
pair matches. The development sweep and frozen settings are retained. The
reference geometry uses box bottoms and a size-scaled pixel error envelope.
That envelope does not estimate occlusion or physical calibration error.

The already-used 1300–1355 diagnostic gave 75 correct and 6 incorrect pair
matches with immediate point linking. Two confirmations gave 61 correct and
zero incorrect. Adding the distance interval gave 57 correct and zero
incorrect. Revocation had no effect on this short diagnostic. IDF1 rose from
33.83% with camera-local IDs to 42.67% with revisable intervals, with sampled
global-ID changes increasing from 10 to 14. WILDTRACK provides shared labels
within its annotated ground region. [WILDTRACK source](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/).

The numeric gate and replay code are frozen before the 1800–1955 holdout.
Same-person pair coverage uses the both-matched denominator and also reports
recall over all labelled pairs. Link-event precision excludes events without
two matchable labels, which are counted separately. Non-overlap transitions
require a sampled disappearance from all previous views and appearance in a
different view within two seconds. Same-view returns require a 0.5–2 s labelled
absence. Missing anchor or destination matches count against recovery. The
public data and sampled transitions bound the scope of the results.

The sealed WILDTRACK replay processed 96 observations across three cameras
and 32 timestamps. All policies used the same original tracker outputs and
fixed crops. The appearance-only legacy arm used the same local track engine.
IDF1 uses online outputs. Pair counts are evaluated among box matches at
IoU .50.

| Policy | IDF1 | Correct shared pairs | Incorrect shared pairs | Sampled global-ID changes |
|---|---:|---:|---:|---:|
| Camera-local | 34.88% | 0 | 0 | 41 |
| Existing shared memory | 35.15% | 0 | 2 | 43 |
| Earlier floor and appearance | 39.39% | 61 | 4 | 47 |
| Immediate point match | 47.61% | 197 | 35 | 53 |
| Confirmed point match | 42.36% | 125 | 24 | 61 |
| Confirmed interval | 40.61% | 79 | 2 | 50 |
| Revisable interval | 38.92% | 58 | 2 | 52 |

The revisable arm achieved 96.67% pair precision at 21.01% coverage among
both-matched pairs (58/276), or 3.12% recall across all 1,860 labelled pairs.
Five accepted links had two verified endpoints and were correct at acceptance.
Two more accepted links could not be verified. Five links were revoked.
Revocation did not improve the measured false pair count on this holdout and
reduced coverage versus keeping confirmed interval links. Mean CPU update time
was 3.9 ms per three-view bundle after precomputing projected contacts and
embeddings. This excludes capture, projection, inference and rendering.

**Gate C remains open.** The frozen minimums include 95% pair precision, 35%
both-matched pair coverage and ten verifiable accepted links. The latter two
failed. The WILDTRACK holdout contained zero qualifying non-overlap transitions
and zero qualifying same-view returns. Do not deploy the replay policy as the
live identity authority on the basis of these results.

Reproduction from checkpoint `bcde816` with the existing v2 environment and
weights. The frozen policy hashes that checkpoint's runner sources.

```powershell
python -m scripts.download_wildtrack_pilot --identity-holdout
python -m selfwatch_v2.wildtrack_tracking --dataset datasets/wildtrack-identity-holdout-v1 --output runs/wildtrack-identity-holdout-v1
python -m scripts.validate_phase4 evaluate --run runs/wildtrack-identity-holdout-v1 --calibration datasets/wildtrack-identity-holdout-v1 --policy docs/phase4-link-policy.json --output runs/phase4-repeat/holdout.json
```

**Independent handoff and return audit.** The i.c.sens sequence 0000 left
images from views 1 and 2 provide independently published 2D boxes and common
person IDs. The run used every fifth frame, 94 observations and 837 pedestrian
boxes. Maximum paired timestamp difference was 12.9 ms. The detector excluded
no outputs based on the labels, and extra predictions are charged. The audit
compares the existing appearance memory with camera-local owners because this
run lacks a validated common-floor transform for the new overlap policy.
[i.c.sens dataset and labels](https://doi.org/10.25835/0082741).

Existing shared memory scored 30.19% IDF1 versus 29.73% for camera-local IDs.
Both recovered zero of nine labelled view transitions and two of 34 same-view
returns. Two recovered returns had zero sampled delay. The camera-local arm
had an anchor ID for one of nine transitions and six of 34 returns. Matching
all detector boxes before local-tracker filtering increased those counts to
three and 16. Shared memory produced zero correct and four incorrect
cross-view pair matches among the labelled box matches.

The fixed-input local tracker controls isolated a second bottleneck. On
WILDTRACK, 532 of 1,860 labelled cross-view pairs had both detector boxes
matched before tracking, and only 276 had two matched original local owners.
Motion/IoU retained all 532 but increased sampled ID changes from 41 to 161.
The sparse-cadence ByteTrack control retained 186 with 42 changes. On i.c.sens,
motion/IoU recovered six of 34 returns with 220 sampled changes, compared
with the original tracker's two of 34 and 121 changes. The ByteTrack control
recovered one return and output substantially fewer matched boxes. These
controls use the same detections at 2 Hz and do not measure native-rate
performance. The next development target is retaining more observed local
owners without increasing ID switches, followed by an independent handoff
test with sufficient transitions and surveyed topology.

```powershell
python -m scripts.download_icsens_identity datasets/icsens-identity-v1
python -m scripts.validate_icsens_identity datasets/icsens-identity-v1 runs/phase4-icsens-v1
python -m scripts.audit_phase4_coverage --wildtrack runs/wildtrack-identity-holdout-v1 --icsens runs/phase4-icsens-v1 --output runs/phase4-repeat/coverage.json
```

The frozen reports are `docs/phase4-holdout-results.json`,
`docs/phase4-icsens-results.json` and `docs/phase4-coverage-results.json`.
All 162 v2 regression tests passed.

### E26 — local continuity and dormant handoff follow-up, 14 November 2026

The CPU replay compares a motion-and-appearance local tracker with a variant
that preserves observed owners from the original cognitive engine. The
preserving variant can extend a recent owner onto a currently measured
detection when the original tracker leaves it unassigned. It cannot override
an original observed owner or re-emit a detection the engine explicitly
suppressed. The separate dormant handoff candidate uses configured directional
routes, an appearance margin, a bounded time window and repeat confirmation.
If both sides later appear together, it restores the conflicting owner to its
original local identity. The route test uses two public views and has no
surveyed entry or exit zones. The new association code remains a replay arm.

Frozen development used WILDTRACK 1700–1785 and i.c.sens sequence 0000.
Holdout inputs were WILDTRACK 1500–1615 and i.c.sens sequence 0006. The
i.c.sens sequence was selected using annotation availability and episode
counts before scoring any candidate on its images. On WILDTRACK development,
preserving owners retained 201 pairs versus 200 at the same nine sampled
changes. The calibrated overlap arm reached 72 correct and three incorrect
pairs, or 96% precision at 35.82% both-matched coverage. On the i.c.sens
holdout, the selected local motion-and-appearance arm improved IDF1 from
27.73% to 36.68%, reduced changes from 60 to 37, and recovered four of 29
same-view returns versus one. Dormant handoff linking recovered zero of 11
labelled transitions and lowered IDF1 to 35.36%.

The 640 detector supplied an eligible box on both ends of only three of
the 11 sampled handoff episodes under the declared matching window. Even an
oracle association on those fixed detections could therefore recover at most
27.27% of those episodes against the 50% gate. The motion-and-appearance local
arm preserved those three opportunities. This bound concerns the sampled
recording, detector inputs and evaluation window. The project must still
demonstrate recovery with stronger observations and sufficient independent
transitions before enabling the replay policy in the live identity authority.

All 166 v2 regression tests passed, including measured returns, reconnects,
departure-only handoff, ambiguity and prospective alias revocation.

The frozen three-camera holdout completed on 72 observations. IDF1 was
29.35% for the original camera-local IDs and 46.08% for calibrated overlap.
The overlap arm linked 177 of 261 both-matched pairs and produced four
incorrect pair matches, yielding 67.82% coverage and 97.79% precision.
The local preserving arm alone changed IDs 19 times versus 21 for the
original. The overlap arm reported 30 changes, including shared-ID
reassignments. All nine accepted links had verified endpoints at acceptance,
and two links were later revoked. Mean CPU linking time was 2.79 ms per
three-view bundle with cached contacts and appearance features. It excludes
capture, projection, detector, ReID and rendering.

The post-holdout resolution diagnostic increased matched pedestrian boxes
from 283 at 640 to 334 at 960 on i.c.sens 0006. Both resolutions supplied
matched boxes at both ends of exactly three of the 11 handoffs, so this
change alone could not meet the handoff target on this recording.

The overlap arm cleared the 95% pair-precision and 35% pair-coverage floors.
It fell one short of ten verified links. The independent handoff and return
tests had enough labelled episodes but missed their recovery floors. Gate C
therefore remains open and the replay policies have no live promotion.

```powershell
python -m scripts.download_wildtrack_pilot --identity-continuity-holdout
python -m selfwatch_v2.wildtrack_tracking --dataset datasets/wildtrack-phase4b-holdout-v1 --output runs/wildtrack-phase4b-holdout-v1
python -m scripts.download_icsens_identity datasets/icsens-phase4b-holdout-v1 --sequence 6
python -m scripts.validate_icsens_identity datasets/icsens-phase4b-holdout-v1 runs/phase4b-icsens-holdout --sequence 6
python -m scripts.validate_phase4_continuity evaluate --wildtrack runs/wildtrack-phase4b-holdout-v1 --calibration datasets/wildtrack-phase4b-holdout-v1 --icsens runs/phase4b-icsens-holdout --policy docs/phase4-continuity-policy.json --output runs/phase4b-repeat/holdout.json
python -m scripts.audit_phase4_resolution datasets/icsens-phase4b-holdout-v1 runs/phase4b-icsens-holdout runs/phase4b-repeat/endpoint-ceiling.json
```

Retained reports: `docs/phase4-continuity-results.json` and
`docs/phase4-continuity-endpoints.json`. The next evaluation should use
native-rate, labelled transitions with independently verified visibility at
both ends. The supplied public reference geometry covers the overlapping
WILDTRACK views. Disjoint-camera qualification needs documented topology
and sufficient observations at the entry and departure of each route.
