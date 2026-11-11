# Identity study and the next measured changes

Completed 2 November 2026. This stage compares identity policies and establishes a labelled evaluation path. It does not claim research-grade tracking accuracy.

## Completed comparison

`runs/identity-study-pilot-v3/` contains **14 paired retail clips, 848 camera observations and 577 YOLO person detections**. Both views use the original recorded timestamps. These are archive-prefix training samples with irregular selected frames; the highest detected occupancy is three people in a view. They do not constitute a crowded-store benchmark.

Three configurations receive the same cached YOLO observations and OSNet feature vectors:

| Configuration | Assigned detections | Sum of local person tokens across clips |
|---|---:|---:|
| Motion/IoU baseline | 577 | 58 |
| Original cognitive engine | 496 | 24 |
| Experimental tentative weak births | 533 | 27 |

These are **output counts**, not recall, IDF1, verified shopper counts or proof that fewer IDs are better. The baseline assigns every input person detection, including possible false detections. The original engine's lower token count partly reflects dropped births. False merges, false positives and missed people require annotations to distinguish.

The original detector cutoff is 0.25 in the new pipeline, while the inherited tracker normally starts tracks at confidence 0.5. Thus YOLO can detect a person without the tracker giving that detection an ID. This is a separate failure from the detector missing a person entirely.

## Experimental change

The candidate lets unmatched person detections at confidence 0.25–0.5 create internal tentative tracks. Their scores remain unchanged. They receive an outward-facing Person ID only after at least two supporting observations. A missing processed observation removes an unconfirmed weak birth; confirmed identities retain the existing recovery path. Territorial suppression, overlap locks, anchor protection and the crowded-scene birth restriction remain active.

This is an opt-in study configuration, not the application default. Confirmation on sparse samples is not evidence of uninterrupted visibility between them; maximum-gap policies need evaluation on continuous footage.

Two examples:

- Sample `014798`: assigned observations increase from 2/13 to 10/13.
- Sample `016699`: assigned observations increase from 3/15 to 9/15.

Visual inspection of `016699` at 7.20 s also exposes a remaining detector problem: the person is visible in rank0 but has no person detection. The candidate restores an ID for rank1's 0.48-confidence detection, while rank0's previous identity remains held in memory. Tracker changes cannot repair every detector miss. Overhead-person detection and cross-view ReID still need domain evaluation.

The study uses all eligible clean ReID crops, with refresh scheduling disabled **equally** for all configurations. The normal app retains attention scheduling. Hard memory-admission policies remain configuration-specific. This is a configuration ablation, not a claim that every isolated cognitive module has been independently measured.

CPU identity timings are diagnostic only: cases run in a fixed order and later cases can benefit from CPU caches. Shared perception, pose, decoding and output cost are excluded. Do not derive a live FPS or speedup claim from those timings.

## Reproducible inputs

Each original cache contains immutable observation JSONL, feature NPZ, model provenance and SHA-256 checksums. Source videos are checked before cache reuse, and frame IDs, camera epochs, timestamps, geometry and source paths must align exactly. JSON loads never enable pickle. The final study records code hashes and confirms no source changed during processing.

The initial development runs `v1` and `v2` remain archived. Use `v3` for reported results: it includes the weak-singleton expiry fix and hashes the saved cache manifest bytes, avoiding integer-versus-string class-map key sorting differences after JSON reload.

## Review and real identity scores

Open http://127.0.0.1:8767/. Three cases have full images and the same 133-landmark predictions. Use **Tracking policy** to switch configurations without changing the video evidence. Navigation returns to the full results table.

For identity evaluation:

1. Label every visible person box on selected frames. Give the same physical shopper the same manual subject reference across frames and cameras, independently of model IDs.
2. Set the complete-person checkbox and mark each finished frame reviewed. Leave uncertain/incomplete frames unreviewed.
3. Export the review. The evaluator rejects incomplete frames, unnamed people, wrong runs, duplicate owners and inconsistent frame sources. Assistant/pseudo labels require an explicit flag and are not independent ground truth.
4. Score each policy against that same exported file. A study uses one common run ID for its cases.

```powershell
.\.venv\Scripts\python.exe -m selfwatch_v2.evaluate_tracking --run runs/identity-study-pilot-v3/016699 --annotations C:/Users/Dev/Downloads/selfwatch-016699-review.json --case legacy --output runs/identity-original-labelled.json
.\.venv\Scripts\python.exe -m selfwatch_v2.evaluate_tracking --run runs/identity-study-pilot-v3/016699 --annotations C:/Users/Dev/Downloads/selfwatch-016699-review.json --case weak_births --output runs/identity-candidate-labelled.json
```

The implementation computes global identity matching from all IoU-compatible identity pairs, following the identity-metric formulation rather than treating greedy frame matches as identity truth. References: [Ristani et al., identity metrics](https://arxiv.org/abs/1609.01775), [TrackEval implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py). Online scores use only recorded IDs. An explicit retrospective mode applies final aliases and is labelled as using future information.

Reported local/global change diagnostics concern sampled reviewed matches; they are not full-sequence CLEAR IDSW. HOTA is not implemented or reported by this tool. The [HOTA reference](https://arxiv.org/abs/2009.07736) is saved for the subsequent official benchmark integration.

## Next research decisions

The immediate bottleneck is independent identity-labelled overhead/crowded video, together with detector adaptation data. Add held-out identities and stores before selecting thresholds or fine-tuning checkpoints. Then evaluate birth delay, missed-person coverage, false merges, IDF1 and an official HOTA implementation jointly; test full occlusion and cross-camera reappearance separately.

The recent [SCOUT dataset/toolkit](https://github.com/cvlab-epfl/scout_toolkit) is relevant to multi-view ground truth and calibration. Its [download workflow](https://scout.epfl.ch/download) requires personal/institutional registration and terms; no registration, personal-data submission or terms acceptance was performed. Its associated `2507.08494` paper was already in the saved library. The additional IDF1 and HOTA PDFs are now saved and indexed, bringing the library to 30 PDFs.

The research hypothesis remains revisable identity–item memory under uncertainty. The new birth experiment suggests that uncertainty must cover identity creation as well as recovery: suppressing weak person evidence can otherwise remove downstream item events. This needs prior-art comparison and labelled ablations; no novelty or publication result is claimed.

## Commands and checks

```powershell
# Original fixed-input inference/cache pass (new directory required)
.\.venv\Scripts\python.exe -m selfwatch_v2.identity_study --output runs/identity-study-fresh --review-top 3
# CPU identity replay of the existing caches
.\.venv\Scripts\python.exe -m selfwatch_v2.identity_study --output runs/identity-study-replay --cache-root runs/identity-study-pilot-v1 --review-top 0
.\.venv\Scripts\python.exe -m selfwatch_v2.study_review --study runs/identity-study-pilot-v3
.\scripts\open-review.ps1 -ReviewDirectory runs/identity-study-pilot-v3 -Port 8767
```

60 v2 tests and nine original multicamera checks pass. They include cache-only reproducibility, corruption rejection, unchanged scheduled extraction, weak-birth confirmation and singleton expiry, cross-camera ID scoring, identity splits despite perfect boxes, case isolation, and prevention of retrospective alias leakage. `pip check` and `git diff --check` pass. All original run directories and annotation keys were retained.
