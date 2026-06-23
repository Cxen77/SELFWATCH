import re
from collections import Counter, defaultdict

LOG = "trace_ipl.log"
PAT = re.compile(
    r"\[MATURE TRACK LOST\] local=(\d+) hits=(\d+) pred_err=([\d.]+)px "
    r"vel=([\d.]+) max_dist=([\d.]+) area_ratio=([\d.]+) aspect_ratio=([\d.]+) "
    r"reid_sim=([-\d.]+) frozen=(\w+) cd=(\w+)"
)

# Live gates (from strongsort_tracker.py)
MAX_COS = 0.30          # app_dist gate -> reid_sim >= 0.70 passes
APP_PASS = 1.0 - MAX_COS  # 0.70
FROZEN_APP = 0.40       # frozen_app_veto: app_dist>0.40 -> reid_sim<0.60
# Shadow (IdentityStore.would_claim against EMA)
SHADOW_EMA = 0.62

events = []
with open(LOG, "r", encoding="utf-16", errors="ignore") as f:
    for line in f:
        m = PAT.search(line)
        if not m:
            continue
        lid, hits, perr, vel, mdist, ar, asp, sim, fr, cd = m.groups()
        events.append(dict(
            lid=int(lid), hits=int(hits), perr=float(perr), vel=float(vel),
            mdist=float(mdist), ar=float(ar), asp=float(asp), sim=float(sim),
            frozen=(fr == "True"), cd=(cd == "True"),
        ))

n = len(events)
print("TOTAL [MATURE TRACK LOST] events parsed:", n)

def classify(e):
    # Priority order mirrors code gate precedence.
    if e["cd"]:
        return "cooldown_lock"        # C-BIoU hard block (recovery lock)
    if e["frozen"] and e["sim"] < 0.60:
        return "frozen_gate"          # frozen_app_veto (app_dist>0.40)
    if e["frozen"]:
        return "frozen_gate"          # frozen but reid ok -> still hard-blocked from fallback
    if e["perr"] > e["mdist"]:
        return "too_far"
    if e["sim"] < APP_PASS:
        return "appearance_gate"      # bad_app / crowd (reid_sim<0.70)
    # reid ok, within distance, not frozen: nearest det lost to contention/no-overlap
    return "other_contention_or_no_iou"

cause = Counter()
claim_by_cause = defaultdict(lambda: [0, 0])  # cause -> [shadow_would_claim, total]
live_matchable = 0
shadow_claimable = 0
per_track_causes = defaultdict(Counter)
sim_buckets = Counter()

for e in events:
    c = classify(e)
    cause[c] += 1
    per_track_causes[e["lid"]][c] += 1
    # Live matchable: passes appearance AND distance AND not frozen/cd
    lm = (e["sim"] >= APP_PASS and e["perr"] <= e["mdist"]
          and not e["frozen"] and not e["cd"])
    if lm:
        live_matchable += 1
    # Shadow: would_claim via EMA threshold (appearance only; store ignores
    # frozen/distance/crowd gates by design)
    sc = e["sim"] >= SHADOW_EMA
    if sc:
        shadow_claimable += 1
    claim_by_cause[c][1] += 1
    if sc:
        claim_by_cause[c][0] += 1
    # sim distribution
    b = round(e["sim"], 1)
    sim_buckets[b] += 1

print("\n=== CAUSE RANKING (by frequency) ===")
for c, k in cause.most_common():
    print(f"  {c:32s} {k:6d}  ({100.0*k/n:5.1f}%)")

print("\n=== SHADOW would_claim() ACCEPT RATE per cause (reid_sim>=0.62 EMA) ===")
for c, k in cause.most_common():
    yes, tot = claim_by_cause[c]
    print(f"  {c:32s} {yes:6d}/{tot:<6d} ({100.0*yes/max(1,tot):5.1f}%)")

print(f"\nLive-matchable (reid>=0.70 & within dist & not frozen/cd): {live_matchable} ({100.0*live_matchable/n:.1f}%)")
print(f"Shadow-claimable (reid>=0.62):                              {shadow_claimable} ({100.0*shadow_claimable/n:.1f}%)")
print(f"Recoverable by SHADOW but NOT by LIVE:                      {shadow_claimable - live_matchable}")

print("\n=== reid_sim distribution (rounded 0.1) ===")
for b in sorted(sim_buckets):
    print(f"  reid_sim~{b:.1f}: {sim_buckets[b]:6d}")

print("\n=== TOP 10 tracks by lost-event count ===")
track_tot = Counter({lid: sum(cc.values()) for lid, cc in per_track_causes.items()})
for lid, k in track_tot.most_common(10):
    dom = per_track_causes[lid].most_common(1)[0]
    print(f"  local={lid:4d}  events={k:6d}  dominant={dom[0]} ({dom[1]})")

# Distinct death EPISODES (collapse consecutive same-lid runs)
episodes = []
prev = None
for e in events:
    if prev is None or e["lid"] != prev:
        episodes.append(dict(lid=e["lid"], frames=1, first=e, worst=e))
    else:
        episodes[-1]["frames"] += 1
    prev = e["lid"]
print(f"\nDistinct death EPISODES (consecutive runs collapsed): {len(episodes)}")
ep_cause = Counter(classify(ep["first"]) for ep in episodes)
print("Episode cause ranking:")
for c, k in ep_cause.most_common():
    print(f"  {c:32s} {k:5d} ({100.0*k/len(episodes):5.1f}%)")
