# Original project integration — 2 November 2026

The default v2 review now runs the original cognitive identity engine behind YOLO and whole-body perception. It no longer uses the small motion/IoU baseline as its main tracker. The original source remains in this repository; the newer detector, recorded timestamps, ReID controller, CPU/GPU overlap and review/export tools remain available.

## What runs now

| Existing component | Integration and purpose |
|---|---|
| `trackers/strongsort_tracker.py` | Original StrongSORT association, ByteTrack low-confidence stage, observation coasting, buffered IoU, ambiguity handling and IDSR. Uses the project's existing tracker configuration. |
| `trackers/strack.py` | Original motion, anchor features, quality checks and memory locks. |
| `memory/global_identity.py` | Original trajectory-gated proposals, provisional ownership and ACT-R hold logic. Its IDs are protected **camera-local** person tokens in this adapter. |
| `memory/active_memory.py`, `identity_fingerprint.py` | Active identity, diverse embeddings, stable/recent/best views, color/body/motion fingerprints; CPU work. |
| `memory/phantom.py` | Original invisible trajectory cones, direction/speed gates, confidence decay and static-occluder lifetime extension. |
| `memory/warm_memory.py`, `reasoning.py` | Original longer-term galleries, decay, retrieval and resurrection. |
| `memory/occlusion_groups.py` | Overlap locks, separation cooldowns and exit trajectories. |
| `memory/ownership_state.py`, `ownership_arbitration.py` | Ownership states, protection against duplicate owners and display suppression. |
| `memory/identity_store.py`, `metrics.py`, `event_log.py` | Original read-only shadow audit, consistency metrics and per-camera cognitive JSONL logs. |
| `reid/embedding_extractor.py`, `attention.py` | Existing OSNet and attention scheduling, already integrated with bounded GPU batches. |
| `memory/cognitive.py`, `multicam/global_registry.py` | Per-camera appearance galleries and the single outward-facing cross-camera Person-ID registry. |
| `memory/topology.py` | Camera image-grid transition observations; advisory, without claiming calibrated floor coordinates. |
| `memory/contradiction.py` | Duplicate/hijack flags recorded for review; they do not independently rewrite IDs. |
| `memory/gait.py` | Retained and sampled only when sufficiently dense, uniform footage exists. Gait remains advisory. This irregular pilot cannot support stride estimation. |

`selfwatch_v2/legacy_guard.py` adapts the original engine's inputs/outputs. `selfwatch_v2/pipeline.py` schedules it between feature extraction and shared identity association. The old engine receives injected detections/features: an explicit guard raises if it attempts another detector or unscheduled ReID inference. It does not load RF-DETR alongside YOLO.

## Identity behavior

The video label is now **Person 1**, with a smaller local tracker number and detection score. A local tracker number can change while the protected person token and shared Person ID survive. The memory panel shows ACTIVE, TENTATIVE, FROZEN, THINKING, PHANTOM, HELD and WARM states as applicable, including last-seen age.

Lost owners reserve their shared IDs without being counted as visible people. A returning protected local owner reclaims its ID directly; it does not allocate another Person ID first. Appearance alone cannot assign that held owner to a different local fragment in the same camera. Camera epochs isolate reconnects. Initial independent camera visits can still be joined by a logged cross-view appearance revision; these provisional associations remain unvalidated hypotheses.

The original local memory quality gates remain intact. The existing new clean-crop cross-view policy is retained for spatial-only/low-confidence tracking stages; actual overlap locks and sustained anchor vetoes protect both galleries. This distinction preserves the working cross-camera matcher without absorbing crowded crops into memory.

Phantom predictions are memory entries, not solid boxes representing observed people. Raw YOLO detections and predicted landmarks remain distinguishable from tracked identity evidence.

## Necessary fixes

- Preserve original image dimensions when injecting YOLO boxes; the legacy application's default 960×540 resize remains unchanged for its existing callers.
- Use capture time for warm-memory reasoning/decay and a declared 30 Hz reference timebase for the old integer-frame motion/hold rules. Global ownership and occlusion cooldowns advance by the supplied elapsed ticks. These inherited thresholds still require calibration on continuous store footage.
- Use the phantom's **actual** confidence decay when forecasting expiry. Previously, the pipeline assumed `0.97^delta`, while prediction could use `(0.96 × 0.92)^delta`. With a larger gap, a phantom could disappear before being saved to warm memory. Both paths now share the exact calculation, including static-occluder treatment.
- Preserve the canonical FROZEN ownership state after activation, matching the existing freeze policy.
- Version review JS/CSS and fetch recording data without caching, so changing the run on the same port does not show the previous IDs/UI. The launcher replaces only verified project-owned review servers.

## Retained but not enabled blindly

The original `retail/` rules, shelf-change detector, old frontends, TensorRT tooling, evaluators, forensic tracer and historical alternative ownership/deferred-assignment modules remain present. Retail alarms and shelf changes are disabled in this selected-frame review because valid store zones, complete visits and checkout/POS evidence are absent. Alternative modules that directly override IDs are not installed as a second identity authority. The old frontend's full keyboard/debug controls have not all been reproduced in the review UI.

The earlier new motion baseline remains available using `--tracker-engine baseline`; the default is `--tracker-engine legacy`. This supports an actual ablation instead of discarding either implementation.

## Hardware assignment

- GPU: YOLO segmentation/detection and batched FP16 OSNet, called from one scheduling thread.
- CPU: decode, StrongSORT, fingerprints, ownership, phantom/warm memory, association, audits, SQLite/JPEG output and the bounded two-thread whole-body ONNX session.
- One CPU pose worker; at most two pending observations. Offline replay drops no observations. CPU identity logic does not trigger extra GPU model copies.

## Verification and evidence

45 v2 regression tests and the original nine multicamera checks pass. Integration tests execute the real original engine with controlled synthetic embeddings: two crossing people, reordered detections, overlap locks, a distant lookalike, an eight-second loss, loss beyond tracker survival with warm recovery, reconnects, label mapping, sparse-video gait abstention and prevention of hidden inference. These are software regressions, not measured real-world ReID accuracy.

`runs/combined-cognitive-v2-000398/` contains the real 64-observation paired-camera replay, source/model provenance, observations database, per-frame memory/audits, cognitive JSONL logs, timings and reference-model comparison. It records three protected returns and a reciprocal cross-camera link at 5.50 s (cosine 0.7866, unchanged 0.78 threshold); both final views show Person 1. There are 56 person detections/pose results and 54 assigned track observations: low-confidence detections do not automatically create new tracks under the inherited birth policy.

This run took 4.87 seconds after model loading, across 64 selected observations. Cognitive CPU processing averaged 1.11 ms, with 1.99 ms p95. These figures describe this run, not a live multi-camera FPS guarantee. Annotated crowded video, IDF1/HOTA/ID-switch evaluation, calibration, item custody and POS reconciliation remain necessary research work.

Open the latest review from the project root:

```powershell
.\scripts\open-review.ps1
```

Reproduce into a new output directory:

```powershell
.\.venv\Scripts\python.exe -m selfwatch_v2.demo --manifest datasets/retailaction-replay-verified/000398.json --output runs/combined-cognitive-new --allow-overlap
.\.venv\Scripts\python.exe -m unittest discover -s tests_v2 -q
.\.venv\Scripts\python.exe -m tests.test_multicam
```

The current viewer is at http://127.0.0.1:8766/. Earlier run directories and their browser annotation keys were preserved.
