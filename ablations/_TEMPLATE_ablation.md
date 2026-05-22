# Ablation Study: {Variable Name}

**ID**: abl{NNN}
**Date**: YYYY-MM-DD
**Status**: Planned | Running | Completed
**Base Experiment**: exp{NNN}
**Git Commit**: `{hash}`

---

## Objective

What component/parameter is being ablated and why?

## Variable Under Test

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `CONFIG_PARAM` | What it controls | current value |

## Variants

| Variant | Value | Rationale |
|---------|-------|-----------|
| A (control) | {default} | Baseline / current setting |
| B | {value} | {why test this} |
| C | {value} | {why test this} |
| D | {value} | {why test this} |

## Control Conditions

Everything else held constant:
- Dataset: 
- Detector: 
- Other config: Identical to base experiment

## Results

### Comparison Table

| Variant | Value | MOTA | IDF1 | ID Sw. | FPS | Identity Stab. | Notes |
|---------|-------|------|------|--------|-----|---------------|-------|
| A | | | | | | | Control |
| B | | | | | | | |
| C | | | | | | | |
| D | | | | | | | |

### Key Observations

- 

### Statistical Significance

| Comparison | p-value | Significant? |
|-----------|---------|-------------|
| A vs B | | |
| A vs C | | |
| A vs D | | |

## Conclusions

### Winner
Variant {X} with value {Y} because...

### Sensitivity Analysis
How sensitive is performance to this parameter?
- High sensitivity: Small changes cause large metric swings
- Low sensitivity: Performance is robust to this parameter

### Recommendation
Keep current value / Change to {X} / Further investigation needed

## Impact

- [ ] Update `config.py` with recommended value
- [ ] Update `docs/architecture/DESIGN_DECISIONS.md`
- [ ] Reference in paper ablation table
