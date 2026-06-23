import re, io

LOG = "runtime_today.log"

# Read, stripping NULs (file is UTF-16-ish from PowerShell redirect)
with open(LOG, "rb") as f:
    raw = f.read().replace(b"\x00", b"")
text = raw.decode("utf-8", errors="ignore")
lines = text.splitlines()

# ── Patterns ──────────────────────────────────────────────────────────
RE_OWN = re.compile(r"\[OWNERSHIP ASSIGNMENT\] frame=(\d+) local=(\d+) gid=(\d+) hits=(\d+)")
RE_FOR = re.compile(r"\[FORENSIC MATCH\] frame=(\d+) det=\d+ trk=\d+ local=(\d+) ")
RE_MAT = re.compile(r"\[MATURE TRACK LOST\] local=(\d+) hits=(\d+) pred_err=([\d.]+)px vel=([\d.]+) max_dist=([\d.]+) area_ratio=([\d.]+) aspect_ratio=([\d.]+) reid_sim=([-\d.]+) frozen=(\w+) cd=(\w+)")
RE_DIV = re.compile(r"\[SHADOW DIVERGENCE\] live minted NEW gid=(\d+) but store would re-own gid=(\d+) \(sim=([\d.]+), dormant=(\d+)f\)")
RE_SPAWN = re.compile(r"NEW LOCAL TRACK SPAWNED: local=(\d+)")
RE_GIDASSIGN = re.compile(r"GLOBAL NEW GID ASSIGNED: local=(\d+) -> new_gid=(\d+)")
RE_STATUS = re.compile(r"\[STATUS F#(\d+)\].*IDswitches=(\d+)")

# Frame tracking: OWNERSHIP/FORENSIC carry frame=, but MATURE/DIV/SPAWN do not.
# We assign each non-framed line the most-recent seen frame number.
cur_frame = 0

# local -> latest known gid (from OWNERSHIP ASSIGNMENT, the authoritative settled map)
local_to_gid = {}
# local -> latest hits seen (ownership or mature)
local_max_hits = {}

# Records
mature_losses = []   # dict per MATURE TRACK LOST line
divergences = []     # dict per SHADOW DIVERGENCE line
status_switches = [] # (frame, idsw)

for ln in lines:
    mf = re.search(r"frame=(\d+)", ln)
    # update current frame from any line that carries an explicit frame=
    mo = RE_OWN.search(ln)
    if mo:
        fr, lid, gid, hits = int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), int(mo.group(4))
        cur_frame = fr
        local_to_gid[lid] = gid
        local_max_hits[lid] = max(local_max_hits.get(lid, 0), hits)
        continue
    fo = RE_FOR.search(ln)
    if fo:
        cur_frame = int(fo.group(1))
        continue
    ms = RE_STATUS.search(ln)
    if ms:
        cur_frame = int(ms.group(1))
        status_switches.append((int(ms.group(1)), int(ms.group(2))))
        continue
    mm = RE_MAT.search(ln)
    if mm:
        lid = int(mm.group(1)); hits = int(mm.group(2))
        local_max_hits[lid] = max(local_max_hits.get(lid, 0), hits)
        mature_losses.append(dict(
            frame=cur_frame, lid=lid, hits=hits,
            pred_err=float(mm.group(3)), vel=float(mm.group(4)),
            max_dist=float(mm.group(5)), reid_sim=float(mm.group(8)),
            frozen=(mm.group(9) == "True"), cd=(mm.group(10) == "True"),
            gid=local_to_gid.get(lid),  # gid this local currently owns (may be None)
        ))
        continue
    md = RE_DIV.search(ln)
    if md:
        divergences.append(dict(
            frame=cur_frame, new_gid=int(md.group(1)),
            reown_gid=int(md.group(2)), anchor_sim=float(md.group(3)),
            dormant=int(md.group(4)),
        ))
        continue

n_div = len(divergences)
print(f"Parsed: {len(mature_losses)} MATURE-LOST lines, {n_div} divergences, "
      f"{len(status_switches)} status lines")

# ── 1. Correlate: each divergence re-owns gid=D. Find whether the local that
#       owned gid=D had a MATURE-LOST with reid_sim<0.70 in the frames just
#       before the divergence (the EMA-drift fingerprint). ─────────────────
# Build gid -> list of (frame, reid_sim, lid, pred_err, max_dist) mature losses
from collections import defaultdict
mature_by_gid = defaultdict(list)
for m in mature_losses:
    if m["gid"] is not None:
        mature_by_gid[m["gid"]].append(m)

WINDOW = 30  # frames before divergence to look back for the drift event
ema_attributable = []
unmatched_div = []
for d in divergences:
    g = d["reown_gid"]; fr = d["frame"]
    cands = [m for m in mature_by_gid.get(g, [])
             if 0 <= (fr - m["frame"]) <= WINDOW and m["reid_sim"] < 0.70
             and not m["frozen"]]
    # spatial gate must have passed: pred_err <= max_dist
    cands = [m for m in cands if m["pred_err"] <= m["max_dist"]]
    if cands:
        worst = min(cands, key=lambda m: m["reid_sim"])  # lowest live sim
        gap = d["anchor_sim"] - worst["reid_sim"]
        ema_attributable.append(dict(div=d, mat=worst, gap=gap))
    else:
        unmatched_div.append(d)

print("\n=== 1. MATURE failures attributable to EMA drift ===")
print(f"  Divergences with a matching mature-loss (reid_sim<0.70, spatial-pass, not frozen)")
print(f"  within {WINDOW} frames of the same gid: {len(ema_attributable)} / {n_div}")
print(f"  Divergences WITHOUT such evidence:                 {len(unmatched_div)} / {n_div}")

print("\n=== 2. Percentage of total divergences ===")
print(f"  EMA-drift-attributable: {100.0*len(ema_attributable)/n_div:.1f}% of {n_div}")

print("\n=== 3. Top 10 worst EMA-vs-anchor gaps (anchor_sim - live reid_sim) ===")
ranked = sorted(ema_attributable, key=lambda x: -x["gap"])
print(f"  {'gap':>6} {'anchor':>7} {'live':>6} {'gid':>4} {'lid':>4} {'frame':>6} {'pred_err':>8} {'max_d':>6} {'dormant':>7}")
for x in ranked[:10]:
    d, m = x["div"], x["mat"]
    print(f"  {x['gap']:6.3f} {d['anchor_sim']:7.3f} {m['reid_sim']:6.3f} "
          f"{d['reown_gid']:4d} {m['lid']:4d} {d['frame']:6d} "
          f"{m['pred_err']:8.1f} {m['max_dist']:6.0f} {d['dormant']:7d}")

# ── 4. Would anchor matching have prevented the 3 final ID switches? ──────
print("\n=== 4. ID-switch prevention analysis ===")
print("  Final ID switches (from cognitive_events jsonl):")
print("    #1 lid=26 gid 26->21 @ frame~287")
print("    #2 lid=47 gid 32->9")
print("    #3 lid=51 gid 32->9")
# Report all mature losses with low reid_sim for gids 21/9/32/26 near switch frames
for g in (21, 9, 32, 26):
    evs = [m for m in mature_by_gid.get(g, []) if m["reid_sim"] < 0.70]
    print(f"  gid={g}: {len(evs)} mature-loss(reid<0.70) events; "
          f"min reid_sim={min((m['reid_sim'] for m in evs), default=float('nan')):.3f}" if evs
          else f"  gid={g}: 0 mature-loss(reid<0.70) events")

# Global stats on mature losses
all_mat_low = [m for m in mature_losses if m["reid_sim"] < 0.70 and not m["frozen"] and m["pred_err"] <= m["max_dist"]]
print("\n=== CONTEXT: all mature losses with spatial-pass + appearance-fail ===")
print(f"  reid_sim<0.70 & pred_err<=max_dist & not frozen: {len(all_mat_low)} per-frame events")
distinct_lids = sorted(set(m['lid'] for m in all_mat_low))
print(f"  distinct local ids affected: {len(distinct_lids)} -> {distinct_lids}")
