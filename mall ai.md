# Small AI work and handoff

Updated 11 November 2026. SELFWATCH's priority is person tracking across cameras.
Use this file for bounded support work that can continue if the main session
resets. The [master project log](docs/PROJECT_LOG.md) holds research decisions and
measured results.

## Ready for a smaller AI

| Task | Deliverable | How to check |
|---|---|---|
| Check the WILDTRACK pilot artifacts | Verify the file sizes and SHA-256 hashes against `datasets/wildtrack-pilot-v1/manifest.json`. Record the file count and any mismatches here. | Read only. Report missing or mismatched files; preserve all inputs. |
| Check the replay instructions | Verify the saved sample path and document the command to start the existing frontend. | Read the launcher and compare the documented port with its arguments. Do not start a webcam or interrupt a session. |
| Review the three-camera score report | Check that the counts in E19 of the master log agree with `runs/wildtrack-tracking-v1/results.json` and `appearance-funnel.json`. | Report exact differences with file references. Do not change metrics or thresholds. |
| Check new paper records | Verify all six PDFs/texts referenced by `docs/TRACKING_STABILITY_AND_FPS.md` and the research manifest. | Check SHA-256 and non-empty extracted text. Do not execute downloaded repository code. |
| Audit the new FPS report | Compare E21 against the eager-control and graph JSONs. | Keep tracking-only, balanced and detailed separate; do not treat offline throughput as live multi-camera FPS. |

## Completed by smaller AIs

No delegated work has been recorded yet. When a task finishes, add the date,
task, agent name, changed files, exact verification result, and remaining
issues. A proposed task or a partial check is not a completed entry.

## Resume the main work

The geometry experiment completed. Frozen settings and separate temporal
evaluation are in `runs/wildtrack-geometry-policy-v2.json` and
`runs/wildtrack-geometry-evaluation-v2.json`: 51 correct shared pairs, no false
shared pairs, but sampled global ID changes increased 10 to 15. Do not claim
reduced ID switching or install these scene-specific matrices for user videos.
Development data: `runs/wildtrack-tracking-v1/`; evaluation data:
`datasets/wildtrack-geometry-evaluation-v1/`. Both manifests are present.

The main agent added CUDA-graph ReID, warmed batch buckets, and scheduled
optional pose. `docs/TRACKING_STABILITY_AND_FPS.md` holds the new literature
review and next experiments. `runs/crowded-tracking-graph-20261108.json` records
4.30/4.15 detailed, 6.74/6.08 balanced and 10.42/10.06 tracking-only FPS, on one
crowded public camera with 8-17 detections. Identical assignments across modes
are not an accuracy guarantee. Keep these changes and verify them after reset.
No smaller AI has been dispatched; do not list main-agent work as delegated.

Primary-agent checks (8 November): all 72 WILDTRACK development files match
their sizes/SHA-256; all 41 paper PDFs match the research manifest; all six new
paper texts are nonempty. 139 v2 tests passed in 10.165 seconds. The new public
Balanced frontend run `studio-aca7697912adc9a3` completed 64 observations with
one cross-camera link. It is a smoke test, not labelled accuracy evidence.
The local frontend is on port 8768; no webcam was opened for this check.
Final Tracking smoke run `studio-bb009542bd2727ac` also completed 64 observations,
one cross-camera link, with Person 1 on both views. Playback play/pause labels
were checked. The verified screenshot and throughput figure are in
`runs/tracking-performance-20261108/`. Continue with the ordered experiments
in `docs/TRACKING_STABILITY_AND_FPS.md`, not a new detector replacement.
Latest verified recovery checkpoint: `runs/research-checkpoint-20261108-tracking-v2/`.
It contains the source archive, model hashes and selected experiment evidence;
retain existing datasets and weights in place. Phase 2 passed its scoped perception gate on 11 November.

The continuous MOT17 study is recorded in E22 of `docs/PROJECT_LOG.md`. Native
frames and labels are in `datasets/mot17-continuity-native-v1/`. The frozen
margin policy produced 96 switches on the evaluation sequence, down from 122,
and recovered the same ID in 9 of 13 visibility episodes, up from 7. ByteTrack
is a reference using the same detections. It reached 36 switches and 46.31%
HOTA. The next identity experiment compares motion association while retaining
the existing memory and recovery stages. GitHub branch `codex/v2-replay-perception`
holds the source checkpoints. No smaller AI has been dispatched.

The scheduled ReID replay is complete in `runs/mot17-continuity-scheduled-v1/`.
It produced 88 → 79 switches and 39.69% → 40.32% IDF1. The two offline replay
rates were 13.48 and 15.88 observations/s in a single fixed-order comparison.
Use E22 for reproduction commands and compare future results against both the
cached and scheduled runs.

Phase 2 closeout is in E23 of `docs/PROJECT_LOG.md`. Reproduce the frozen
person operating point with `docs/phase2-final-protocol.json` and
`scripts.validate_phase2`. The fresh pair reached 22/24 recall targets with
zero false positives and 7.3–8.0 processed FPS per view. Stage 3 is next.

Phase 3 reference rig completed on 11 November. E24 in
`docs/PROJECT_LOG.md` records the CPU geometry and timing contract, 36
verified WILDTRACK frames, 17 usable manually marked contacts, 10 zone
and abstention checks and 156 passing tests. Keep the marked contacts and
frozen JSON report. Store calibration needs measured floor targets, shared
timing events and installed camera metadata. Use the geometry evidence in
Phase 4 to test links and false merges before enabling it in the live tracker.
No smaller AI was dispatched for this milestone.

Phase 4 research checkpoint is recorded in E25 of `docs/PROJECT_LOG.md`.
The new overlap-memory replay preserves the local owner and can revoke a
link prospectively. Use the frozen protocol and retain both false-link counts
and coverage. No smaller AI was dispatched.

The sealed WILDTRACK report and independent i.c.sens handoff audit are now
recorded in E25. The revisable policy reached 96.67% pair precision but only
21.01% both-matched coverage. Existing shared memory recovered 0/9 i.c.sens
view transitions and 2/34 same-view returns. Gate C remains open. The coverage
controls show why the next run should focus on observed local owners and ID
stability before cross-camera tuning. Retain the frozen policy and results.
