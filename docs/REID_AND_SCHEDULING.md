# Reusing identity memory and overlapping CPU and GPU work

Baseline implemented 1 November 2026. **The default has since moved to the [combined original-engine integration](OLD_PROJECT_INTEGRATION.md)**, served at http://127.0.0.1:8766/ from `runs/combined-cognitive-v2-000398/`. The baseline run below is retained for comparison. The earlier observation-only demo remains separate on port 8765 when its server is running.

## Actual reuse

- The original `reid.embedding_extractor.EmbeddingExtractor` supplies OSNet features, pinned staging buffers, batched inference and its dedicated CUDA stream. The v2 adapter uses 256×128 crops and FP16 CUDA, while preserving the old constructor's 128×128 default.
- The original `GlobalMultiCameraIdentityManager` is the sole global-ID allocator and mapping owner. It now accepts an injected capture clock and accounts separately for unique global IDs and per-view assignments. Dormant age uses that clock too. Default wall-clock behavior still supports the old application.
- The original `CognitiveMemory` supplies stable/recent/best-quality embeddings and diversity galleries, separately for each camera. The v2 adapter controls admission and expiration; it does not run a competing global allocator.
- The original `CognitiveAttention` supplies high/normal/low priorities. Refresh intervals use capture time rather than modulo arithmetic on unevenly sampled frames. New metadata eviction releases expired tracklet schedules.

The old dormant-only cross-camera matcher is not used unchanged: the new controller also searches identities active in another view when overlap is explicitly enabled. This baseline initially omitted the full tracker; the linked combined integration now restores it. Retail alarm rules and TensorRT engines remain in the repository and are not enabled in the selected-frame review.

## Weights and numerical checks

The author-hosted OSNet x1.0 MSMT17-combineall checkpoint was downloaded from pinned Hugging Face revision `a5c5cc037c24235cda3b21085b93ad77c9616224`. File SHA-256: `48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727`. Source/size/hash are recorded in `weights/osnet/provenance.json`.

Strict validation matched all 565 feature tensors; only classifier weight/bias are ignored. Missing feature tensors now raise in strict mode. Loading uses `weights_only=True`; crop shape and batch-capacity errors are rejected before inference. Two numerical fixture crops produced 512-dimensional normalized embeddings with CPU/FP16-GPU cosine agreement of approximately 0.99987 and 0.99986. This verifies numerical consistency, not person identity accuracy. See `runs/reid-validation.json`.

## Association policy

Global assignments are appearance hypotheses, not verified identities or calibrated probabilities. Defaults: cosine threshold 0.78, ambiguity margin 0.06, two fresh supporting observations, active-track timeout 2 seconds and gallery search horizon 300 seconds. An overlap flag must explicitly permit simultaneous views.

Close lookalikes abstain from merging. Two independently observed tracks in the same camera cannot share an identity. Unavailable/low-confidence or substantially overlapping person crops do not generate appearance vectors. A provisional ID may be revised to another identity with a recorded old/new link event. A group already linked across cameras is protected from automatic group-to-group merging; a large appearance conflict is flagged and excluded from gallery updates.

The development trace initially recovered a fragmented track within one camera but did not consistently pass cross-view confirmation. The policy was then extended to admit reciprocal strong observations from different cameras within a three-second window spanning at least 150 ms, at the same threshold and margin. This is an engineering adjustment made using this development clip, not a validated generalization claim. It needs held-out multi-person lookalike trials.

Current real replay trace: at 5.50 seconds rank1 local track 1 changes from provisional G2 to G1 using reciprocal-view support. At 9.30 seconds rank1 local track 2 changes from G3 to G1. The last recorded frames of both cameras therefore use G1. Source camera and association kind are logged, so same-camera recoveries are not blindly counted as cross-camera events. Ambiguous candidates and local IDs remain visible. Link history in the UI only reveals events available by the selected replay time.

Galleries are capped per identity/view, and at most 200 identity profiles are retained before inactive profiles are evicted. IDs are never recycled. The durable replay observations, frame associations and revision trace are saved; automatic restoration of identity galleries across sessions, calibrated world geometry and item-custody corrections are still later stages.

## Scheduling implementation

One caller owns GPU detector and ReID inference. One bounded CPU worker owns the RTMW session. The main pipeline can prepare the next observation while CPU pose processes the preceding one, with at most two jobs in flight. Outputs commit in capture order. Replay drops no frames; worker failures propagate. The current JPEG/SQLite export still executes in the commit path, so it has not been disguised as cost-free asynchronous output.

CPU tasks: decode, local tracking, galleries/cosine association, memory and pose. GPU tasks: YOLO and scheduled OSNet batches. Neural pose placement is still a bottleneck to compare against GPU/optimized/selective inference. Simply putting all possible operations on the CPU would not guarantee better latency.

Artifacts include per-frame detector, ReID, tracking, association, CPU-pose, pose-queue, processing-age and image-write times. The observed peak PyTorch reserved allocation in these runs was 154 MiB; this excludes CUDA context/driver allocations and other GPU processes and is not a total VRAM measurement.

## Measurements and limitations

Standalone replay timings varied considerably: an early overlapped run took 5.24 seconds, another took 17.13; serial runs took 19.26 and 6.73 seconds. These process-to-process figures are not a defensible speedup comparison.

A follow-up benchmark alternated serial/parallel execution twice each in one warmed process, using the same 64 frames and model instances and checking identical frame/global-ID assignments:

| Mode | Trial seconds | Aggregate processed frames/s | Median processing age |
|---|---|---|---|
| Serial | 17.25, 16.67 | 3.71, 3.84 | 270, 271 ms |
| Overlapped | 13.75, 13.86 | 4.65, 4.62 | 419, 434 ms |

This run observed roughly 23% greater aggregate compute throughput, accompanied by higher processing age from queueing. It includes decode and compute, excludes JPEG/SQLite export and model construction, and uses preselected offline frames. It is not a three-camera live FPS result. Only approximately 0.53 GiB of 15.64 GiB system RAM was available during an accompanying system check; background workload and memory pressure limit benchmark stability. No unrelated user processes were stopped. Detailed trials are in `runs/overlap-benchmark.json`.

The next optimization should compare pose placement and selective crop scheduling, while tracking p95 age and memory alongside throughput. Live input requires an explicit bounded stale-frame policy rather than unbounded pose work.

## Verification and reproduction

37 v2 tests and all nine original multicamera infrastructure checks passed. Tests include overlapping/disjoint association, same-camera exclusivity, ambiguous matches, reconnect epochs, event-time dormant age, feature checkpoint completeness, bounded GPU/CPU overlap, ordered results and worker errors. Numerical fixtures validate CPU/GPU embedding agreement.

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/download_osnet.py
.\.venv\Scripts\python.exe -m selfwatch_v2.demo --manifest datasets/retailaction-replay-verified/000398.json --output runs/new-reid-review --allow-overlap
.\.venv\Scripts\python.exe -m scripts.benchmark_overlap --manifest datasets/retailaction-replay-verified/000398.json
.\.venv\Scripts\python.exe -m scripts.validate_reid --image runs/paired-review-000398-v2/images/000032.jpg
.\.venv\Scripts\python.exe -m unittest discover -s tests_v2 -v
.\.venv\Scripts\python.exe -m tests.test_multicam
```

Use `--serial` for the sequential pipeline comparison. Use a fresh output directory for each artifact run. The user-facing review remains a prototype: shared memory is present, but cross-store/long-gap ReID, dense-crowd accuracy, POS reconciliation and reliable unpaid-exit decisions remain unvalidated.
