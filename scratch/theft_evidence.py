import sys, re
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

log_file = r'C:\Users\Dev\Desktop\SELFWATCH\anchor_run.log'
with open(log_file, 'r', encoding='utf-16', errors='replace') as f:
    lines = f.readlines()

mat_losts = {}
for i, line in enumerate(lines):
    l = line.strip()
    if 'MATURE TRACK LOST' not in l:
        continue
    m = re.search(r'local=(\d+) hits=(\d+) pred_err=([\d.]+)px vel=([\d.]+) max_dist=([\d.]+).*reid_sim=([\d.]+)', l)
    if m:
        mat_losts[i+1] = {
            'local': int(m.group(1)), 'hits': int(m.group(2)),
            'pred_err': float(m.group(3)), 'max_dist': float(m.group(5)),
            'reid_sim': float(m.group(6)),
        }

print('=== CONCRETE DETECTION THEFT EVIDENCE ===')
print('Cases: lost track has reid>=0.55 AND pred_err<=max_dist, but NO OA assignment for it')
print()

shown = 0
for mat_ln, mat in mat_losts.items():
    if mat['reid_sim'] < 0.55:
        continue
    if mat['pred_err'] > mat['max_dist']:
        continue
    local_in_oa = False
    competing_oa = []
    for ln2 in range(mat_ln - 15, mat_ln + 15):
        if ln2 <= 0 or ln2 > len(lines):
            continue
        ll = lines[ln2-1].strip()
        if 'OWNERSHIP ASSIGNMENT' not in ll:
            continue
        mo = re.search(r'local=(\d+).*det=(\d+).*cost=([\d.]+)', ll)
        if mo:
            lid_oa = int(mo.group(1))
            det_oa = int(mo.group(2))
            cost_oa = float(mo.group(3))
            if lid_oa == mat['local']:
                local_in_oa = True
                break
            competing_oa.append({'local': lid_oa, 'det': det_oa, 'cost': cost_oa, 'ln': ln2})
    if not local_in_oa and competing_oa:
        shown += 1
        if shown > 15:
            continue
        lid = mat['local']
        hits = mat['hits']
        pred = mat['pred_err']
        reid = mat['reid_sim']
        print(f'LOST local={lid} hits={hits} pred_err={pred:.1f}px reid_sim={reid:.3f} [Mat L{mat_ln}]')
        for coa in competing_oa[:4]:
            print(f'  COMPETING: local={coa["local"]} det={coa["det"]} cost={coa["cost"]:.3f} [L{coa["ln"]}]')
        print()

print(f'--- (showing first 15 of many such frames) ---')
