import sys, re
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

log_file = r'C:\Users\Dev\Desktop\SELFWATCH\anchor_run.log'
with open(log_file, 'r', encoding='utf-16', errors='replace') as f:
    lines = f.readlines()

# --- Parse all relevant events ---
events = []
for i, line in enumerate(lines):
    l = line.strip()
    tag = None
    if 'SHADOW DIVERGENCE' in l:     tag = 'DIVERGENCE'
    elif 'OWNERSHIP ASSIGNMENT' in l: tag = 'OWN_ASSIGN'
    elif 'MATURE TRACK LOST' in l:   tag = 'MATURE_LOST'
    elif 'TRACK ENTERED LOST' in l:  tag = 'ENTERED_LOST'
    elif 'NEW LOCAL TRACK SPAWNED' in l: tag = 'SPAWN'
    elif 'GLOBAL NEW GID ASSIGNED' in l: tag = 'GID_ASSIGN'
    elif 'RECOVERED!' in l:          tag = 'RECOVERED'
    elif 'TRACK DIED' in l:          tag = 'DIED'
    elif 'STATUS F#' in l:           tag = 'STATUS'
    if tag:
        events.append((i+1, tag, l))

divs         = [(ln,l) for ln,et,l in events if et=='DIVERGENCE']
own_assigns  = [(ln,l) for ln,et,l in events if et=='OWN_ASSIGN']
mature_losts = [(ln,l) for ln,et,l in events if et=='MATURE_LOST']
entered_losts= [(ln,l) for ln,et,l in events if et=='ENTERED_LOST']
spawns       = [(ln,l) for ln,et,l in events if et=='SPAWN']
gid_assigns  = [(ln,l) for ln,et,l in events if et=='GID_ASSIGN']
recoveries   = [(ln,l) for ln,et,l in events if et=='RECOVERED']
statuses     = [(ln,l) for ln,et,l in events if et=='STATUS']

# --- Build local->gid map (use LAST known gid from OWN_ASSIGN) ---
lid_to_gid = {}
for ln, l in own_assigns:
    m = re.search(r'local=(\d+) gid=(\d+)', l)
    if m:
        lid_to_gid[int(m.group(1))] = int(m.group(2))

# Build gid->lid reverse (use first seen)
gid_to_lid = {}
for lid, gid in lid_to_gid.items():
    if gid not in gid_to_lid:
        gid_to_lid[gid] = lid

# Also parse GID_ASSIGN for new tracks
for ln, l in gid_assigns:
    m = re.search(r'local=(\d+) ->\s*new_gid=(\d+)', l)
    if m:
        lid, gid = int(m.group(1)), int(m.group(2))
        if lid not in lid_to_gid:
            lid_to_gid[lid] = gid

# --- Parse MATURE_LOST per local_id ---
mat_by_local = defaultdict(list)
for ln, l in mature_losts:
    m = re.search(
        r'local=(\d+) hits=(\d+) pred_err=([\d.]+)px vel=([\d.]+) max_dist=([\d.]+)'
        r' area_ratio=([\d.]+) aspect_ratio=([\d.]+) reid_sim=([\d.]+) frozen=(\w+) cd=(\w+)', l)
    if m:
        mat_by_local[int(m.group(1))].append({
            'ln': ln, 'hits': int(m.group(2)),
            'pred_err': float(m.group(3)), 'vel': float(m.group(4)),
            'max_dist': float(m.group(5)), 'reid_sim': float(m.group(8)),
            'frozen': m.group(9)=='True', 'cd': m.group(10)=='True',
        })

# --- Parse OWNERSHIP ASSIGNMENT per frame ---
# [OWNERSHIP ASSIGNMENT] frame=N local=L gid=G hits=H det=D iou=I app=A cdist=C cost=CO ...
oa_by_frame = defaultdict(list)
for ln, oa_l in own_assigns:
    m = re.search(
        r'frame=(\d+) local=(\d+) gid=(\d+) hits=(\d+) det=(\d+) iou=([\d.-]+) app=([\d.-]+) cdist=([\d.]+) cost=([\d.]+)',
        oa_l)
    if m:
        oa_by_frame[int(m.group(1))].append({
            'ln': ln,
            'frame': int(m.group(1)), 'local': int(m.group(2)),
            'gid': int(m.group(3)), 'hits': int(m.group(4)),
            'det': int(m.group(5)), 'iou': float(m.group(6)),
            'app': float(m.group(7)), 'cdist': float(m.group(8)),
            'cost': float(m.group(9)),
        })

# --- Determine frame number from STATUS lines ---
# STATUS F#N -> line number
frame_to_ln = {}
for ln, l in statuses:
    m = re.search(r'STATUS F#(\d+)', l)
    if m:
        frame_to_ln[int(m.group(1))] = ln

# Build line -> frame approximation
# Find the nearest STATUS before a given line
ln_list = sorted(frame_to_ln.keys(), key=lambda f: frame_to_ln[f])
def approx_frame_for_line(target_ln):
    """Return (frame, line_of_status) closest before target_ln."""
    best_f, best_ln = None, 0
    for f in ln_list:
        sln = frame_to_ln[f]
        if sln <= target_ln:
            best_f, best_ln = f, sln
        else:
            break
    return best_f, best_ln

# --- E_RECOVERY classification ---
# Skip F_IMMEDIATE (dormant=1, sim>=0.95)
e_recovery_divs = []
for div_ln, div_text in divs:
    m = re.search(r'gid=(\d+) but store would re-own gid=(\d+) .sim=([\d.]+), dormant=(\d+)f', div_text)
    if not m:
        continue
    new_gid, old_gid, sim, dormant = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
    if dormant == 1 and sim >= 0.95:
        continue
    e_recovery_divs.append((div_ln, new_gid, old_gid, sim, dormant))

print(f'E_RECOVERY divergences to analyze: {len(e_recovery_divs)}')
print()

# --- For each E_RECOVERY divergence, analyze ---
summary = {'A_STOLEN': 0, 'B_GATE_REJECT': 0, 'C_NO_DETECTION': 0, 'D_COST_COMP': 0, 'E_OTHER': 0}
causal_chains = []

for idx, (div_ln, new_gid, old_gid, sim, dormant) in enumerate(e_recovery_divs):
    # 1. Find old track's local_id
    old_lid = gid_to_lid.get(old_gid)

    # 2. Get MATURE_LOST for old_lid, just before the divergence
    mat_for_old = [d for d in mat_by_local.get(old_lid, []) if d['ln'] <= div_ln]

    # 3. Get last few MATURE_LOST entries (most recent before divergence)
    mat_recent = mat_for_old[-5:] if mat_for_old else []
    mat_last = mat_for_old[-1] if mat_for_old else None

    # 4. Find approximate frame for the divergence
    approx_f, _ = approx_frame_for_line(div_ln)

    # 5. Look at OWNERSHIP ASSIGNMENTs in frames just before and around the divergence
    oa_in_window = []
    for f, oas in oa_by_frame.items():
        for oa in oas:
            if div_ln - 400 <= oa['ln'] <= div_ln + 20:
                oa_in_window.append(oa)

    # 6. Did the lost track have a nearby-detection with reid >= 0.35?
    candidate_exists = mat_last is not None and mat_last['reid_sim'] >= 0.35
    candidate_within_spatial = mat_last is not None and mat_last['pred_err'] <= mat_last['max_dist']
    candidate_strong = candidate_exists and mat_last['reid_sim'] >= 0.55

    # 7. Classify
    if mat_last is None:
        cat = 'C_NO_DETECTION'
        reason = 'No MATURE_LOST found for this old_gid->local mapping'
    elif mat_last['reid_sim'] < 0.35 and not candidate_within_spatial:
        cat = 'B_GATE_REJECT'
        reason = f'reid_sim={mat_last["reid_sim"]:.3f} < 0.35 AND pred_err={mat_last["pred_err"]:.1f} > max_dist={mat_last["max_dist"]:.1f}'
    elif mat_last['reid_sim'] < 0.35:
        cat = 'B_GATE_REJECT'
        reason = f'reid_sim={mat_last["reid_sim"]:.3f} < 0.35 (bad_app_gate)'
    elif not candidate_within_spatial:
        cat = 'B_GATE_REJECT'
        reason = f'pred_err={mat_last["pred_err"]:.1f} > max_dist={mat_last["max_dist"]:.1f} (spatial gate)'
    else:
        # Candidate exists: reid>=0.35, within spatial gate
        # Was this detection consumed by another track in OA logs near the same frame?
        # OA logs only show MATURE tracks (hits>100). Look for same det index claimed by another track.
        # We can't know exact det index, but we can check if the OA window has many competing tracks
        n_competing = len(oa_in_window)
        if n_competing > 0:
            cat = 'A_STOLEN'
            reason = (f'reid_sim={mat_last["reid_sim"]:.3f} OK, pred_err={mat_last["pred_err"]:.1f} <= max_dist={mat_last["max_dist"]:.1f}; '
                      f'{n_competing} OA events in window — detection consumed by competing tracks')
        else:
            cat = 'D_COST_COMP'
            reason = f'reid_sim={mat_last["reid_sim"]:.3f} OK, within spatial gate, no OA events found in window'

    summary[cat] = summary.get(cat, 0) + 1

    # Build causal chain
    chain = {
        'idx': idx+1, 'div_ln': div_ln, 'new_gid': new_gid, 'old_gid': old_gid,
        'sim': sim, 'dormant': dormant, 'old_lid': old_lid,
        'cat': cat, 'reason': reason,
        'mat_last': mat_last,
        'n_competing_oa': len(oa_in_window),
        'oa_sample': oa_in_window[-3:] if oa_in_window else [],
    }
    causal_chains.append(chain)

print('=== SUMMARY ===')
total = len(e_recovery_divs)
for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
    pct = 100*count/max(1,total)
    label = {
        'A_STOLEN': 'A) Detection stolen by competing track',
        'B_GATE_REJECT': 'B) Detection rejected by gate',
        'C_NO_DETECTION': 'C) Detection never existed',
        'D_COST_COMP': 'D) Hungarian cost competition (no OA evidence)',
        'E_OTHER': 'E) Other',
    }.get(cat, cat)
    print(f'  {label}: {count}  ({pct:.1f}%)')

print()
print('=== CAUSAL CHAINS (all events) ===')
for ch in causal_chains:
    print(f'\n[{ch["idx"]:02d}] L{ch["div_ln"]} | new_gid={ch["new_gid"]} stole old_gid={ch["old_gid"]} | sim={ch["sim"]:.3f} dormant={ch["dormant"]}f')
    print(f'     old_lid={ch["old_lid"]}  cat={ch["cat"]}')
    print(f'     reason: {ch["reason"]}')
    if ch['mat_last']:
        ml = ch['mat_last']
        print(f'     Last MATURE_LOST [L{ml["ln"]}]: pred_err={ml["pred_err"]:.1f}px max_dist={ml["max_dist"]:.1f}px reid_sim={ml["reid_sim"]:.3f} vel={ml["vel"]:.1f}')
    if ch['oa_sample']:
        print(f'     Competing OA events ({ch["n_competing_oa"]} total in window):')
        for oa in ch['oa_sample']:
            print(f'       [L{oa["ln"]}] frame={oa["frame"]} local={oa["local"]} gid={oa["gid"]} det={oa["det"]} iou={oa["iou"]:.3f} app={oa["app"]:.3f} cdist={oa["cdist"]:.1f}px cost={oa["cost"]:.3f}')
