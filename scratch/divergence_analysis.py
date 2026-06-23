import sys, re
sys.stdout.reconfigure(encoding='utf-8')

log_file = r'C:\Users\Dev\Desktop\SELFWATCH\anchor_run.log'
with open(log_file, 'r', encoding='utf-16', errors='replace') as f:
    lines = f.readlines()

events = []
for i, line in enumerate(lines):
    l = line.strip()
    if 'SHADOW DIVERGENCE' in l:
        events.append((i+1, 'DIVERGENCE', l))
    elif 'NEW LOCAL TRACK SPAWNED' in l:
        events.append((i+1, 'SPAWN', l))
    elif 'TRACK ENTERED LOST STATE' in l:
        events.append((i+1, 'ENTERED_LOST', l))
    elif 'MATURE TRACK LOST' in l:
        events.append((i+1, 'MATURE_LOST', l))
    elif 'TRACK DIED' in l:
        events.append((i+1, 'DIED', l))
    elif 'RECOVERED!' in l:
        events.append((i+1, 'RECOVERED', l))
    elif 'ANCHOR DRIFT' in l:
        events.append((i+1, 'ANCHOR_DRIFT', l))
    elif 'IDENTITY VETO' in l:
        events.append((i+1, 'IDENTITY_VETO', l))
    elif 'BIRTH SUPPRESSED' in l:
        events.append((i+1, 'BIRTH_SUPPRESS', l))

divs = [(ln, l) for ln, et, l in events if et == 'DIVERGENCE']
spawns = [(ln, l) for ln, et, l in events if et == 'SPAWN']
entered_losts = [(ln, l) for ln, et, l in events if et == 'ENTERED_LOST']
mature_losts = [(ln, l) for ln, et, l in events if et == 'MATURE_LOST']
dieds = [(ln, l) for ln, et, l in events if et == 'DIED']
recoveries = [(ln, l) for ln, et, l in events if et == 'RECOVERED']
anchor_drifts = [(ln, l) for ln, et, l in events if et == 'ANCHOR_DRIFT']
identity_vetos = [(ln, l) for ln, et, l in events if et == 'IDENTITY_VETO']
birth_sups = [(ln, l) for ln, et, l in events if et == 'BIRTH_SUPPRESS']

print('=== MATURE TRACK LOST STATISTICS ===')
mat_data = []
for ln, l in mature_losts:
    pat = r'local=(\d+) hits=(\d+) pred_err=([\d.]+)px vel=([\d.]+) max_dist=([\d.]+) area_ratio=([\d.]+) aspect_ratio=([\d.]+) reid_sim=([\d.]+) frozen=(\w+) cd=(\w+)'
    m = re.search(pat, l)
    if m:
        mat_data.append({
            'ln': ln,
            'local': int(m.group(1)),
            'hits': int(m.group(2)),
            'pred_err': float(m.group(3)),
            'vel': float(m.group(4)),
            'max_dist': float(m.group(5)),
            'area_ratio': float(m.group(6)),
            'aspect_ratio': float(m.group(7)),
            'reid_sim': float(m.group(8)),
            'frozen': m.group(9)=='True',
            'cd': m.group(10)=='True',
        })

total = len(mat_data)
print(f'Total MATURE_LOST samples: {total}')
unique_trks = len(set(d['local'] for d in mat_data))
print(f'Unique tracks logged: {unique_trks}')
print()

# Nearest-detection reid_sim bins
sim_bins = [(0.0,0.35),(0.35,0.50),(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,1.01)]
print('reid_sim (nearest-det) distribution:')
for lo,hi in sim_bins:
    cnt = sum(1 for d in mat_data if lo<=d['reid_sim']<hi)
    print(f'  [{lo:.2f},{hi:.2f}): {cnt}  ({100*cnt/max(1,total):.1f}%)')

print()
print('pred_err > max_dist (beyond spatial gate):',
      sum(1 for d in mat_data if d['pred_err'] > d['max_dist']),
      f'({100*sum(1 for d in mat_data if d["pred_err"] > d["max_dist"])/max(1,total):.1f}%)')
print()

# Classify: appearance-only failure vs spatial-only vs combined vs fine-but-still-lost
fail_app_only   = [d for d in mat_data if d['reid_sim'] < 0.35 and d['pred_err'] <= d['max_dist']]
fail_spat_only  = [d for d in mat_data if d['reid_sim'] >= 0.35 and d['pred_err'] > d['max_dist']]
fail_both       = [d for d in mat_data if d['reid_sim'] < 0.35 and d['pred_err'] > d['max_dist']]
fine            = [d for d in mat_data if d['reid_sim'] >= 0.35 and d['pred_err'] <= d['max_dist']]
print('Failure mode for nearest-detection-when-mature-lost:')
print(f'  Appearance only failure (reid<0.35, within dist):  {len(fail_app_only)} ({100*len(fail_app_only)/max(1,total):.1f}%)')
print(f'  Spatial only failure (reid>=0.35, beyond dist):    {len(fail_spat_only)} ({100*len(fail_spat_only)/max(1,total):.1f}%)')
print(f'  Both failed (reid<0.35 AND beyond dist):           {len(fail_both)} ({100*len(fail_both)/max(1,total):.1f}%)')
print(f'  FINE (reid>=0.35 AND within dist) -- other cause:  {len(fine)} ({100*len(fine)/max(1,total):.1f}%)')

print()
print('=== DIVERGENCE DORMANT FRAME ANALYSIS ===')
dormants = []
sims_div = []
for ln, l in divs:
    md = re.search(r'dormant=(\d+)f', l)
    ms = re.search(r'sim=([\d.]+)', l)
    if md:
        dormants.append(int(md.group(1)))
    if ms:
        sims_div.append(float(ms.group(1)))

print(f'Total divergences: {len(divs)}')
print(f'Dormant range: min={min(dormants)} max={max(dormants)} avg={sum(dormants)/len(dormants):.1f}')
print(f'Similarity range: min={min(sims_div):.3f} max={max(sims_div):.3f} avg={sum(sims_div)/len(sims_div):.3f}')
print()
one_frame = sum(1 for d in dormants if d == 1)
short_term = sum(1 for d in dormants if 2 <= d <= 20)
medium = sum(1 for d in dormants if 21 <= d <= 100)
long_term = sum(1 for d in dormants if d > 100)
print(f'dormant=1 (new-birth same frame as lost):   {one_frame}  ({100*one_frame/len(divs):.1f}%)')
print(f'dormant=2-20 (short recovery-window miss):  {short_term}  ({100*short_term/len(divs):.1f}%)')
print(f'dormant=21-100 (medium miss):               {medium}  ({100*medium/len(divs):.1f}%)')
print(f'dormant>100 (long / track-expired miss):    {long_term}  ({100*long_term/len(divs):.1f}%)')

# sim=1.000 means the new local_track essentially IS the same as what the store had
# These are immediate re-births where the tracker failed to recover the lost track
high_sim = sum(1 for s in sims_div if s >= 0.95)
print(f'\nsim>=0.95 (near-identical re-birth):  {high_sim}  ({100*high_sim/len(divs):.1f}%)')
print(f'sim<0.65 (weaker association):         {sum(1 for s in sims_div if s < 0.65)}  ({100*sum(1 for s in sims_div if s < 0.65)/len(divs):.1f}%)')

print()
print('=== RECOVERY FUNNEL ===')
print(f'ENTERED_LOST events:    {len(entered_losts)}')
print(f'RECOVERED events:       {len(recoveries)}')
print(f'TRACK DIED events:      {len(dieds)}')
print(f'SPAWN events:           {len(spawns)}')
print(f'SHADOW DIVERGENCE:      {len(divs)}')

# How many spawns were preceded immediately by a divergence?
div_lns = set(ln for ln, _ in divs)

# Classify each divergence event by its primary cause
print()
print('=== DIVERGENCE CLASSIFICATION ===')
# Categories:
# A) Appearance rejection: sim < 0.35 of nearest det to old track -> fused gate kills it
# B) Spatial rejection: pred_err > max_dist -> no candidate within gate distance
# C) Motion prediction failure: frozen track (frozen=True in MATURE_LOST near spawn)
# D) Lost-track expiration: old_gid's local track DIED before divergence
# E) Recovery-window miss: dormant > max_lost (but tracker alive, divergence because GlobalIDMapper waited)
# F) Track state / tentative transition (sim=1.0, dormant=1, track confirmed too late)
# G) Immediate re-birth (dormant=1, sim~1.0, same detection -> tracker created new local before recovery)

# From the analysis, find old_gid's DIED entries
all_deaths_by_local = {}
for ln, l in dieds:
    m = re.search(r'local=(\d+)', l)
    if m:
        lid = int(m.group(1))
        all_deaths_by_local.setdefault(lid, []).append(ln)

classification = {}
for idx, (div_ln, div_text) in enumerate(divs):
    m = re.search(r'gid=(\d+) but store would re-own gid=(\d+) \(sim=([\d.]+), dormant=(\d+)f\)', div_text)
    if not m:
        classification[idx] = 'H'
        continue
    new_gid, old_gid, sim, dormant = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))

    # Rule 1: dormant=1, sim>=0.95 -> immediate re-birth (track lost then spawned on same detection)
    if dormant == 1 and sim >= 0.95:
        classification[idx] = 'F_IMMEDIATE'
        continue

    # Rule 2: old local track has DIED before this divergence -> D (Lost-track expiration)
    # Note: old_gid == global_id, not local_id. We look for died tracks near the old_gid
    old_died = old_gid in all_deaths_by_local and any(ln < div_ln for ln in all_deaths_by_local[old_gid])
    if old_died and dormant > 100:
        classification[idx] = 'D_EXPIRED'
        continue

    # Rule 3: short dormant (2-20f) -> recovery window miss: Stage 1B failed
    # Determine if appearance or spatial is the primary rejection
    # Look at MATURE_LOST for old_gid near div_ln
    mat_old = [d for d in mat_data if d['local'] == old_gid and d['ln'] < div_ln]
    mat_old_near = [d for d in mat_old if div_ln - d['ln'] < 200]
    if mat_old_near:
        last_mat = mat_old_near[-1]
        if last_mat['pred_err'] > last_mat['max_dist'] and last_mat['reid_sim'] >= 0.35:
            cat = 'B_SPATIAL'
        elif last_mat['reid_sim'] < 0.35:
            cat = 'A_APPEARANCE'
        elif last_mat['reid_sim'] < 0.55 and last_mat['pred_err'] <= last_mat['max_dist']:
            cat = 'A_APPEARANCE_BORDERLINE'
        elif last_mat['frozen']:
            cat = 'C_FROZEN_SPATIAL'
        else:
            cat = 'E_RECOVERY'
    else:
        # No MATURE_LOST for old_gid -> track not in mature logging but still lost
        if dormant > 100:
            cat = 'D_EXPIRED'
        elif dormant <= 5:
            cat = 'G_RAPID_LOSS'
        else:
            cat = 'E_RECOVERY'
    classification[idx] = cat

from collections import Counter
cats = Counter(classification.values())
print('Category counts:')
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')

# Print representatives for each category
print()
print('=== CATEGORY REPRESENTATIVES ===')
cat_examples = {}
for idx, cat in classification.items():
    cat_examples.setdefault(cat, []).append(idx)

for cat in sorted(cat_examples.keys()):
    print(f'\n--- {cat} ---')
    for i in cat_examples[cat][:2]:
        div_ln, div_text = divs[i]
        print(f'  [{i+1}] L{div_ln}: {div_text}')

print()
print('=== ANCHOR DRIFT STATS ===')
print(f'Total ANCHOR_DRIFT events: {len(anchor_drifts)}')
if anchor_drifts:
    ema_vals = []
    anc_vals = []
    gap_vals = []
    for ln, l in anchor_drifts:
        me = re.search(r'ema=([\d.]+)', l)
        ma = re.search(r'anchor=([\d.]+)', l)
        mg = re.search(r'gap=([\d.]+)', l)
        if me: ema_vals.append(float(me.group(1)))
        if ma: anc_vals.append(float(ma.group(1)))
        if mg: gap_vals.append(float(mg.group(1)))
    if ema_vals:
        print(f'EMA sim avg: {sum(ema_vals)/len(ema_vals):.3f}  min: {min(ema_vals):.3f}')
        print(f'Anchor sim avg: {sum(anc_vals)/len(anc_vals):.3f}')
        print(f'Gap avg: {sum(gap_vals)/len(gap_vals):.3f}  max: {max(gap_vals):.3f}')

print()
print('=== IDENTITY VETO STATS ===')
print(f'Total IDENTITY_VETO events: {len(identity_vetos)}')
print(f'Total BIRTH_SUPPRESS events: {len(birth_sups)}')
