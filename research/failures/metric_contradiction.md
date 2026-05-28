# Failure Analysis: Metric Contradiction Between Internal and Visual Systems

Date: 2026-05-23

## Failure Description

The pipeline reported "ID switches: 0, Tracking Continuity: 100%"
while the visual evaluator reported "Visible ID switches: 22, Duplicate
box frames: 125."

Users saw near-perfect metrics while observing visually unstable tracking.

## Root Cause

Two independent metric systems measured different things:

**Pipeline metrics (`self.metrics`):**
- Measured INTERNAL association consistency
- Counted rebinding events in `_local_to_global` mapping
- Scope: identity manager layer only

**Visual evaluator (`SELFWATCHEvaluator`):**
- Measured VISIBLE rendering output
- Counted on-screen identity changes in the final display dict
- Scope: post-arbitration visual layer

Both were correct for their scope. But the system exported both under
similar labels ("ID switches" vs "Visible ID changes") without clearly
distinguishing what each measured.

## Impact

Users trusted the "100% continuity" metric and assumed the system was
working perfectly, when visual evidence showed significant instability.
This delayed diagnosis of the duplicate box problem by several iterations.

## Fix

1. Pipeline close now explicitly labels:
   "INTERNAL TRACKER METRICS (association-level)"
   "NOTE: These measure internal consistency, NOT visual output."

2. Central ownership state machine provides a third, canonical view
   of identity state that can be cross-referenced against both.

3. Future: evaluator metrics should be the PRIMARY reported metrics
   since they measure what the human actually sees.

## Lesson

When a system has multiple metric sources measuring overlapping phenomena,
the export pipeline MUST explicitly label what each metric measures and
what it does NOT measure. Default to measuring the USER-FACING output.
