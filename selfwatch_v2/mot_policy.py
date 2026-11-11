"""Select on the development sequence only; retain baseline when gates fail."""
import argparse
import json
from pathlib import Path
from .mot_continuity import digest


def select(results):
    base=results['baseline']['summary'];eligible=[]
    for case,result in results.items():
        if case=='baseline':continue
        row=result['summary']
        if (row['IDSW']<base['IDSW'] and row['IDF1']>=base['IDF1']-.005 and
                row['HOTA']>=base['HOTA']-.005 and row['CLR_Re']>=base['CLR_Re']-.01):
            eligible.append(case)
    return min(eligible,key=lambda c:(results[c]['summary']['IDSW'],-results[c]['summary']['IDF1'],c)) if eligible else 'baseline'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.study/'results.json').read_text());config=json.loads((args.cache/'cache.json').read_text())
    if report['role']!='development' or report['sequence']!='MOT17-02-FRCNN' or config['role']!='development':raise ValueError('Selection requires predeclared development data')
    expected={'baseline','clean_memory','strict_margin','clean_margin'}
    if set(report['results'])!=expected:raise ValueError('Incomplete development comparison')
    selected=select(report['results'])
    folders=['engine','memory','trackers','multicam']
    paths=[p for folder in folders for p in Path(folder).rglob('*.py')]
    paths.extend(Path('selfwatch_v2')/name for name in ('contracts.py','legacy_guard.py','continuity_guard.py','identity.py','pipeline.py','mot_metrics.py','mot_continuity.py','mot_policy.py','tracking.py'))
    paths.append(Path('config.py'))
    signature={key:config[key] for key in ('input_wh','detector_imgsz','detector_confidence','feature_policy')}
    signature.update(detector_sha256=config['detector']['sha256'],reid_sha256=config['reid']['sha256'])
    policy=dict(schema_version=1,selected_case=selected,development_report_sha256=digest(args.study/'results.json'),
        development_cache_sha256=digest(args.cache/'cache.json'),perception_signature=signature,
        tracker_sources={p.as_posix():digest(p) for p in sorted(set(paths))},
        rule='Strictly fewer IDSW; IDF1/HOTA drops <=0.005 absolute; recall drop <=0.01. Tie: fewer IDSW, then greater IDF1, then case name. Otherwise baseline.',
        thresholds=dict(clean_confidence=.7,clean_max_person_iou=.3,strict_margin=.03),
        live_promotion=False,scope='Frozen preset development selection; separate single-camera sequence evaluation still required')
    with args.output.open('x',encoding='utf-8') as out:json.dump(policy,out,indent=2)
    print('Frozen selected case:',selected,flush=True)

if __name__=='__main__':main()
