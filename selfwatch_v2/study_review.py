"""Build a review hub without changing recorded identity assignments."""
import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import shutil

from .demo import write_review_assets


def build(study):
    report=json.loads((study/'study.json').read_text(encoding='utf-8'))
    selected=report['review_samples']
    navigation=[dict(sample=s,href=f'../{s}/') for s in selected]
    for sample in selected:
        target=study/sample
        data=json.loads((target/'data.json').read_text(encoding='utf-8'))
        if not all('image' in f for f in data['frames']):
            raise ValueError('Selected review lacks source images')
        data['study_navigation']=navigation
        (target/'data.json').write_text(json.dumps(data,allow_nan=False),encoding='utf-8')
        write_review_assets(target)
        for name in ('LICENSE','NOTICE'):
            source=Path('datasets/retailaction-pilot')/name
            if source.exists():shutil.copyfile(source,target/name)
    rows=[]
    for sample in report['samples']:
        cases=sample['cases'];sid=sample['sample']
        assigned=' / '.join(str(cases[key]['assigned_person_detections']) for key in ('baseline','legacy','weak_births'))
        tokens=' / '.join(str(cases[key]['distinct_local_tokens']) for key in ('baseline','legacy','weak_births'))
        review=f'<a href="{escape(sid)}/">Review {escape(sid)}</a>' if sid in selected else '<span class="muted">Cached inputs</span>'
        rows.append(f'<tr><td>{escape(sid)}</td><td>{sample["max_detected_people_per_view"]}</td><td>{cases["baseline"]["person_detections"]}</td><td>{assigned}</td><td>{tokens}</td><td>{review}</td></tr>')
    count=sum(s['observations'] for s in report['samples'])
    cards=[]
    for sid in selected:
        s=next(s for s in report['samples'] if s['sample']==sid)
        c,w=s['cases']['legacy'],s['cases']['weak_births']
        cards.append(f'<a class="card" href="{escape(sid)}/"><span>PAIRED CAMERAS / {escape(sid)}</span><strong>{s["max_detected_people_per_view"]} peak person detections</strong><p>Assigned detections: original {c["assigned_person_detections"]}; experimental {w["assigned_person_detections"]}.</p><b>Open comparison →</b></a>')
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SELFWATCH identity study</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f1f5f7;color:#173143;font:15px/1.6 "Segoe UI",sans-serif}header{padding:23px 5vw;border-bottom:1px solid #d4dfe7;background:white;display:flex;justify-content:space-between}header strong{letter-spacing:.1em}header span{color:#617788}main{max-width:1300px;margin:auto;padding:45px 5vw}h1{font-size:38px;line-height:1.14;max-width:740px;margin:0 0 18px}h2{margin-top:38px;font-size:23px}.lede{font-size:18px;max-width:760px;color:#466275}.stats{display:flex;gap:34px;flex-wrap:wrap;margin:26px 0}.stats b{font-size:25px;display:block}.stats span{color:#607586}.notice{border-left:4px solid #bc821c;padding:12px 18px;background:#fff8e8;max-width:900px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:28px 0}.card{background:white;border:1px solid #d4dfe7;border-top:4px solid #306e9b;border-radius:7px;padding:22px;text-decoration:none;color:inherit}.card span{font-size:11px;letter-spacing:.08em;color:#6a7d8b}.card strong{display:block;font-size:21px;line-height:1.3;margin-top:12px}.card p{font-size:14px;color:#536c7e}.card b{color:#285f8c}.table-wrap{overflow:auto;background:white;border:1px solid #d4dfe7;border-radius:6px}table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:13px 17px;text-align:left;border-bottom:1px solid #e2e9ef}th{background:#e7eff4;font-size:12px}a{color:#255f8d}.muted{color:#748795}footer{margin-top:28px;color:#5c7384;font-size:13px}@media(max-width:850px){.cards{grid-template-columns:1fr}h1{font-size:30px}header{gap:12px;flex-wrap:wrap}}
    </style></head><body><header><strong>SELFWATCH</strong><span>Identity research / development comparison</span></header><main><h1>Measure continuity.<br>Keep missed people in view.</h1><p class="lede">Compare the motion baseline, the original cognitive engine and an experimental tentative-birth policy on identical camera observations.</p>'''
    html+=f'<div class="stats"><div><b>{len(report["samples"])}</b><span>paired clips</span></div><div><b>{count}</b><span>camera observations</span></div><div><b>3</b><span>tracking policies</span></div></div>'
    html+='<p class="notice"><strong>Accuracy is not measured yet.</strong> Fewer IDs can mean fewer detected people receive tracks. Compare assignment coverage alongside continuity, then label persistent subjects to evaluate IDF1.</p>'
    html+='<div class="cards">'+''.join(cards)+'</div><h2>All development clips</h2><p>Triples are baseline / original / experimental. Counts describe model outputs, not ground-truth recall or unique shoppers.</p><div class="table-wrap"><table><thead><tr><th>Sample</th><th>Peak / view</th><th>Person detections</th><th>Assigned B / C / W</th><th>Local tokens B / C / W</th><th>Evidence</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    html+='<footer><p>Selected, irregular RetailAction training frames. Three cases are prepared for visual review with the same body/hand landmarks. The experimental policy is not the application default.</p><p><a href="REPORT.md">Development report</a> · <a href="study.json">Run provenance and diagnostics</a></p></footer></main></body></html>'
    (study/'index.html').write_text(html,encoding='utf-8')
    (study/'review_build.json').write_text(json.dumps(dict(study_sha256=sha256((study/'study.json').read_bytes()).hexdigest(),
        renderer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),samples=selected),indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();build(args.study)


if __name__=='__main__':main()
