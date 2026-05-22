# Failure Case Registry

> Index of all documented failure cases in SELFWATCH. Triage new failures from `logs/forensic/` into `forensics/active/`.

---

## Active Failures (Unresolved)

| ID | Category | Severity | Date | Summary | Dir |
|----|----------|----------|------|---------|-----|
| | | | | | |

## Resolved Failures

| ID | Category | Severity | Date Found | Date Fixed | Root Cause | Fix | Dir |
|----|----------|----------|-----------|-----------|-----------|-----|-----|
| | | | | | | | |

## Failure Statistics

| Category | Active | Resolved | Total |
|----------|--------|----------|-------|
| ID Switch | 0 | 0 | 0 |
| False Resurrection | 0 | 0 | 0 |
| Phantom Mismatch | 0 | 0 | 0 |
| Duplicate ID | 0 | 0 | 0 |
| Identity Hijack | 0 | 0 | 0 |
| Fragmentation | 0 | 0 | 0 |
| Other | 0 | 0 | 0 |

## Raw Forensic Data

Auto-captured failures in `logs/forensic/` follow this naming:
```
fail_{unix_timestamp}_{sequence_num}_{clip|frame|meta}.{mp4|jpg|json}
```

### Triage Workflow
1. Check `logs/forensic/` for new failure captures
2. Review the `_meta.json` to understand the failure
3. Create `forensics/active/FNNN_{description}/` using template
4. Copy relevant evidence files
5. Write root cause analysis
6. When fixed, move to `forensics/resolved/`
