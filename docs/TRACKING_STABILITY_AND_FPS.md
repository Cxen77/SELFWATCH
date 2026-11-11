# Person-ID stability and FPS investigation

Updated 8 November 2026. Current phase remains Stage 2. Geometry results are an
offline prototype; full retail identity continuity is still unvalidated.

## What was measured

The public WILDTRACK C1 stress profile uses the same 12 images, resized to
1280 x 720, with 8-17 detected people (mean 13.3), original 2 Hz source times,
YOLO26n-seg at 640, OSNet, original cognitive memory and identical identity
thresholds. Timing includes SQLite writes, JPEG encoding and JSON serialization;
images are predecoded. It excludes camera capture, browser transfer/display and
final review export. This is processed-observation throughput, not live camera
capacity or an accuracy benchmark. CPU tests ran during the first scheduled
experiment; use the later isolated controls for the principal comparison.

The later eager control achieved 3.56 FPS with detailed landmarks/masks and
7.19 FPS in tracking-only mode. Pose averaged 109 ms; ReID 68 ms; detector
55 ms; cognitive memory plus association 9 ms in detailed mode. The initial
profile also exposed first-use batch-shape stalls: first detailed trial 0.78 FPS,
then 3.82 FPS after those shapes had run. Preserve that cold result rather than
mixing it with warm capacity.

With CUDA graph ReID, repeated trials achieved:

| Mode | Trial FPS | What is computed |
|---|---|---|
| Detailed | 4.30, 4.15 | Same detector, ReID, memory, mask outlines and up to four poses per frame |
| Balanced | 6.74, 6.08 | Same tracking, no mask outlines; up to two poses per view every 0.5 source seconds |
| Tracking | 10.42, 10.06 | Same detector, ReID and memory; pose and mask outlines omitted |

All graph trials and eager controls produced the same local/global-ID assignment
signature on these frames, `c06354b42b812e3735a50980172886befa1e6b7c2f3cbec9a84076086b6e17dd`.
This establishes agreement with the existing tracker on this input, not correct
identity labels. Every source observation is processed. Balanced rotates current
detection indices; it does not claim a fixed per-person landmark sampling rate.
Skipped frames contain no stale landmarks. At 2 Hz every frame is due for pose;
higher-rate sources may skip optional pose calls between those times.

CUDA graph validation uses actual public person crops, five timed calls per
shape after two warmups. Real inputs are copied before replay. Changed-input
tests pass at batches 1, 2, 4, 8 and 16 with maximum absolute embedding error 0.
Padding to warmed batch sizes changes the tested embeddings by at most 3e-8.
See `runs/reid-graph-validation-20261108.json` for current timing and runtime.
This reuses GPU execution sequences, not old embeddings. It follows the fixed
shape, persistent-buffer and single-owner constraints in the
[PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs).
The tested runtime is PyTorch 2.11.0+cu128 on the RTX 4060 Laptop GPU.

The frontend defaults to Tracking and offers Balanced and Detailed explicitly.
CPU handles capture/decode, association, phantom/warm memory, floor matching,
storage and UI delivery. GPU handles YOLO and OSNet, plus optional CUDA pose.
CPU pose remains supported with a bounded worker. Moving neural inference to
CPU indiscriminately would need measurement; it is not inherently faster.
Use `--no-reid-graph` when comparing the original execution path.

Regenerate the paper-ready throughput figure and source-hashed summary with
`python -m scripts.summarize_tracking_performance`. Artifacts are under
`runs/tracking-performance-20261108/`; the portal screenshot shows the separate,
easier paired sample and must not be confused with the crowded benchmark.

## What the new papers contribute

Six complete primary PDFs and extracted texts are retained in
`Desktop/SELFWATCH_RESEARCH/tracking-20261108/` and `papers/`; the manifest now
contains 41 verified PDFs. Read method and experiment sections, not just abstracts.

| Paper | Useful mechanism and decision for SELFWATCH |
|---|---|
| [OA-SORT, CVPR 2026](https://arxiv.org/abs/2603.06034) | Models occlusion from projected depth ordering and box coverage, refines it with Gaussian weights, adjusts positional costs and low-score motion updates. Compare its occlusion-aware update against current phantom/ownership protection. Floor-facing assumptions must be checked for fisheye/top-down cameras. It adds CPU work; do not copy published FPS across hardware. |
| [TrackTrack, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) | Iterative reciprocal minimum-cost association and track-aware birth suppression reduce spurious track creation. The full method also uses detections removed by NMS. Our current detector contract does not supply those; a partial adaptation must be labelled as such. Compare original memory plus conservative birth control on identical detections. |
| [ARC-SORT, October 2026 preprint](https://arxiv.org/abs/2610.22306) | Uses predictive uncertainty for association tolerance and a crossing permutation margin to defer identity decisions. Useful for bounded two-person hypotheses after crossings. Its offline tracklet merge uses future information and cannot be reported as causal live recovery. Code was promised, not available in the paper's abstract when reviewed. Sports results do not establish store transfer. |
| [On-the-Fly Homographies Calibration, September 2026 preprint](https://arxiv.org/abs/2609.18582) | Refines coarse floor mappings using strict cross-view correspondences and centroid agreement. Relevant to reducing geometry-induced false links. Optimization is asynchronous on a separate machine in the described deployment; its zero critical-path latency is not zero total computation. Proprietary evaluation data limit reproduction. Keep surveyed anchors and reject drift/collapse. |
| [BoostTrack++, 2025 revision](https://arxiv.org/abs/2408.13003) | Soft confidence boosting, richer shape/motion/overlap similarity and age-dependent thresholds recover weak detections. Evaluate retained-track recovery separately from new births. Preserve raw detector confidence in evidence; any boosted score is an association score, not a calibrated detection probability. |
| [Deep OC-SORT, 2023 baseline](https://arxiv.org/abs/2302.11813) | Confidence-dependent appearance EMA limits contaminated gallery updates; row/column separation adapts appearance weighting. Relevant baseline for our clean-gallery admission and ambiguity margins. This is older foundational work, not presented as a recent paper. Camera-motion compensation adds little for truly fixed, calibrated cameras unless they move. |

Reviewed official [TrackTrack source](https://github.com/kamkyu94/TrackTrack) at
`ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da` and
[BoostTrack source](https://github.com/vukasin-stanojevic/BoostTrack) at
`fb5bfc3a8f067476565e753b3a73df4d757c9d03`, particularly initialization,
association and confidence-boost logic. Source copies and hashes are saved in
`repository-sources.json`. No third-party training/inference code was executed.

## Geometry result and remaining ID problem

Development selected radius 0.75 m, separation margin 0.25 m and corroborating
appearance cosine 0.55, with two adjacent synchronized sightings. Those settings
were frozen before scoring a different time segment of the same WILDTRACK scene.
Development shared-pair precision was 66/(66+1), or 98.5%. Evaluation yielded
51 correct shared pairs, zero false shared pairs, out of 108 same-person pairs
matched in both views and 494 possible labelled pairs. Diagnostic IDF1 improved
from 33.8% to 41.3%. Geometry alone produced 73 correct and 2 false shared pairs
on evaluation, but failed the development precision gate and was not selected.

This does **not** establish fewer temporal ID switches: sampled global-ID changes
were 10 for the appearance/local control and 15 for geometry, including changes
when camera-local provisional IDs join an earlier global ID. Sampled local-ID
changes stayed 10. These counts are not official full-sequence CLEAR IDSW. Do not
hide the difference using final aliases or rewrite past predictions. The prototype
also lacks bounded long-run union retention and same-camera fragment relinking,
and has not been promoted to live camera policy. Published projected boxes and
an outdoor 2 Hz scene limit both geometry and tracking conclusions.

## Next experiments, in order

1. Preserve current timing/identity outputs. Repeat a longer isolated 1/2/3-camera
   crowd test, including cold load, source frame age, per-camera processing FPS,
   p50/p95 latency, dropped live frames, actual GPU memory and browser overhead.
   Keep the detailed-mode control and report omitted optional work explicitly.
2. Freeze a separately labelled continuous crossing/occlusion development split.
   Compare existing StrongSORT/phantom protection with confidence-aware motion,
   reciprocal association and birth suppression, changing one component at a time.
   Measure IDF1, official IDSW/HOTA via a verified evaluator, fragmentation,
   false merges and recovery delay. Preserve missed people in denominators.
3. Add a bounded ambiguity record: competing owners, clean evidence by camera,
   last-seen time, exclusion constraints and expiry. An ambiguous crossing cannot
   update stable galleries. Publish provisional status while collecting evidence;
   distinguish live resolution from later corrections. Test identity ownership
   collisions and similar-clothing people before UI promotion.
4. Build actual-camera floor calibration and synchronization checks. Project only
   valid visible foot contacts, carry calibration uncertainty, restrict overlap
   links to plausible world distance/time, and use camera-transition priors for
   non-overlapping views. Do not apply WILDTRACK matrices to uploaded footage.
5. Evaluate view-aware ReID prototypes or domain fine-tuning only after clean
   subject/scene-disjoint labels exist. Spend extra embeddings on ambiguous,
   recovered and newly entering tracks; prioritize clean observations without
   starving quiet views. Retain the current full-evidence baseline.

Research hypothesis: uncertainty in identity memory can allocate scarce neural
inference while preserving cross-camera continuity. The proposed contribution
would need a joint quality/compute Pareto result and ablations against these
papers. Combining existing methods alone does not establish novelty.
