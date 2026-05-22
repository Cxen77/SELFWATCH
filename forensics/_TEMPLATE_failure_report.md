# Failure Report: {Short Description}

**ID**: F{NNN}
**Date Detected**: YYYY-MM-DD
**Status**: Active | Investigating | Root-Caused | Resolved
**Severity**: Critical | Major | Minor
**Category**: ID Switch | False Resurrection | Phantom Mismatch | Duplicate ID | Identity Hijack | Fragmentation | Other

---

## Summary

One-paragraph description of the failure.

## Reproduction

### Steps to Reproduce
1. 
2. 
3. 

### Frequency
- Always / Intermittent / Rare
- Approximate rate: X per Y frames

### Conditions
- Scene type: Crowded / Sparse / Doorway / Occlusion-heavy
- Number of people: 
- Lighting: 
- Camera angle: 

## Evidence

### Video Clip
- File: `evidence/clip.mp4`
- Frame range: {start} — {end}

### Key Frame
- File: `evidence/frame.jpg`
- Annotations: {description of what's wrong}

### Metadata
- File: `evidence/meta.json`
- Key fields: 

### Cognitive Event Log
- File: `evidence/events.jsonl`
- Relevant events: 

## Root Cause Analysis

### Immediate Cause
What directly caused the failure?

### Contributing Factors
What conditions made this failure possible?

### System Component
Which module(s) are responsible?
- [ ] Detector
- [ ] Tracker
- [ ] ReID
- [ ] Cognitive Memory
- [ ] Phantom Tracker
- [ ] Contradiction Detector
- [ ] Identity Fingerprint
- [ ] Other: 

## Fix

### Proposed Solution
- 

### Implementation
- File(s) changed: 
- Config changes: 
- Git commit: 

### Verification
How was the fix verified?
- [ ] Failure no longer reproduces
- [ ] No regression on benchmarks
- [ ] Added regression test: `tests/regression/test_{name}.py`

## Lessons Learned

What should we do differently to prevent this class of failure?

## Related

- Similar failures: F{NNN}
- Pattern: `forensics/patterns/{name}.md`
- Experiment: exp{NNN}
