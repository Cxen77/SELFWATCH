# Ablation Registry

> Master index of all SELFWATCH ablation studies. Update this file whenever a new ablation is created.

---

## Completed Ablations

| ID | Variable | # Variants | Date | Key Finding | Link |
|----|----------|-----------|------|-------------|------|
| | | | | | |

## Planned Ablations

| ID | Variable | Variants to Test | Priority | Motivation |
|----|----------|-----------------|----------|-----------|
| abl001 | Fusion Weights | 5 combinations | High | Validate 6-weight balance |
| abl002 | ReID Skip Rate | 4 intervals | Medium | FPS vs accuracy tradeoff |
| abl003 | Phantom Max Age | 30/60/90/120 frames | Medium | Occlusion bridge duration |
| abl004 | Memory Gallery Size | 3/5/8/12 | Low | Embedding storage tradeoff |
| abl005 | Detector Resolution | 320/384/448/512 | High | Speed vs detection quality |
| abl006 | Gait Signal Weight | 0/0.05/0.10/0.20 | Medium | Gait contribution value |
| abl007 | Topology Grid Size | 4/8/12/16 | Low | Spatial granularity |
| abl008 | Confidence Threshold | 0.35/0.45/0.55/0.65 | High | Detection sensitivity |

---

## Naming Convention

```
abl{NNN}_{variable_being_ablated}/
```
