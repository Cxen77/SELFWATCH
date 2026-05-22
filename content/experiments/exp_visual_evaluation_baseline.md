# Experiment: SELFWATCH Visual Evaluation Baseline

Date: 2026-05-23

## Configuration

- Detector: RT-DETR Nano
- Resolution: 384
- ReID: OSNet x1.0
- FP16: Enabled
- torch.compile: Enabled
- Fast Preprocess: Enabled

## Runtime Performance

- Runtime: 70.7s
- Frames Processed: 1283
- Average FPS: 18.1

## Profiling

| Subsystem | Avg Time |
|---|---|
| Detection | 21.03 ms |
| ReID | 17.40 ms |
| Tracker | 2.17 ms |
| Memory | 1.78 ms |
| Draw | 1.49 ms |

## Human-Perceived Evaluation Metrics

| Metric | Value |
|---|---|
| Visible ID Switches | 22 |
| Teleportations | 11 |
| Duplicate Box Frames | 125 |
| Identity Stability | 90.2% |

## Internal Tracker Metrics

| Metric | Value |
|---|---|
| Internal ID Switches | 0 |
| Tracking Continuity | 100% |

## Key Insight

The new evaluation framework successfully exposed hidden visual failures ignored by traditional internal MOT metrics.