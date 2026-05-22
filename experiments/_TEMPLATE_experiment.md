# Experiment: {Experiment Name}

**ID**: exp{NNN}
**Date**: YYYY-MM-DD
**Author**: 
**Status**: Planned | Running | Completed | Failed | Abandoned
**Git Commit**: `{hash}`
**Git Branch**: `exp/exp{NNN}-{name}`
**Parent Experiment**: exp{NNN-1} (or "None" if first)

---

## Objective

What is this experiment trying to determine?

## Hypothesis

> Clear, falsifiable hypothesis. Link to `research/hypotheses/HNNN.md` if applicable.

## Configuration Changes

What configuration parameters were changed from the parent experiment?

```yaml
# Changed parameters (relative to parent):
parameter_name: new_value  # was: old_value
```

**Frozen config**: See `config.yaml` in this directory.

## Dataset

| Dataset | Sequences | Frames | Notes |
|---------|-----------|--------|-------|
| | | | |

## Methodology

Step-by-step description of what was done.

1. 
2. 
3. 

## Results

### Quantitative

| Metric | Parent Exp | This Exp | Delta | Significant? |
|--------|-----------|----------|-------|--------------|
| MOTA | | | | |
| IDF1 | | | | |
| HOTA | | | | |
| ID Switches | | | | |
| FPS | | | | |
| Identity Stability | | | | |
| Tracking Continuity | | | | |
| Resurrection Accuracy | | | | |

### SELFWATCH-Specific Metrics

| Metric | Parent Exp | This Exp | Delta |
|--------|-----------|----------|-------|
| Resurrections | | | |
| False Resurrections | | | |
| Retrieval Attempts | | | |
| Retrieval Success Rate | | | |
| Memory Saves | | | |
| Phantom Activations | | | |
| Lock Events | | | |
| Duplicate Box Frames | | | |

### Qualitative Observations

- 

## Analysis

### What Worked
- 

### What Didn't Work
- 

### Surprising Findings
- 

## Negative Findings

### What Didn't Work (Document for Future Reference)
- **Approach**: 
- **Expected**: 
- **Actual**: 
- **Root Cause**: 
- **Lesson**: 

## Conclusions

Summary of findings and their implications.

## Next Steps

- [ ] Follow-up experiment: exp{NNN+1}
- [ ] Update architecture: `docs/architecture/`
- [ ] Update paper: `paper/drafts/`

## Reproduction

```bash
# Commands to reproduce this experiment
git checkout {commit_hash}
python main.py --config experiments/exp{NNN}/config.yaml
```
