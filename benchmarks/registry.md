# Benchmark Registry

> Tracking SELFWATCH performance across datasets, versions, and hardware.

---

## Standard Benchmarks

| Dataset | Latest MOTA | Latest IDF1 | Latest HOTA | Latest FPS | Date | Experiment |
|---------|-------------|-------------|-------------|------------|------|-----------|
| Custom (webcam) | — | — | — | 18.6 | 2026-05-22 | current |
| MOT17 | — | — | — | — | — | — |
| MOT20 | — | — | — | — | — | — |

## SELFWATCH-Specific Metrics (Latest)

| Metric | Value | Date |
|--------|-------|------|
| Identity Stability | 0.958 | 2026-05-22 |
| Tracking Continuity | 1.0 | 2026-05-22 |
| ID Switches | 0 | 2026-05-22 |
| Resurrection Accuracy | 0.021 | 2026-05-22 |
| Retrieval Success Rate | 0.093 | 2026-05-22 |
| Fragmentation Count | 0 | 2026-05-22 |

## Hardware Baselines

| GPU | Detector | Resolution | FPS | Date |
|-----|----------|-----------|-----|------|
| RTX 4060 | RF-DETR nano | 384 | 18.6 | 2026-05-22 |

## Speed Benchmarks

See `speed/fps_history.csv` for full history.

---

## Evaluation Protocol

1. **Dataset**: Specify exact sequences used
2. **Metrics**: MOTA, IDF1, HOTA (via TrackEval), SELFWATCH-specific metrics
3. **Hardware**: Document GPU, CUDA version, batch size
4. **Reproducibility**: Link to frozen config and git commit
