"""Summarize a matched paired-view functional check, without inventing accuracy."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import statistics


def summarize(runs):
    signatures=[];result={}
    for name,root in runs.items():
        data=json.loads((root/'data.json').read_text());summary=json.loads((root/'summary.json').read_text())
        signatures.append([(f['frame']['camera_id'],f['frame']['stream_epoch'],f['frame']['frame_index'],
            f['frame']['captured_at_s'],f['frame']['source'],f['frame']['width'],f['frame']['height']) for f in data['frames']])
        counts=Counter();scores=[]
        for frame in data['frames']:
            people=[p for p in frame['instances'] if p['label'].lower()=='person']
            scores.extend(p['score'] for p in people)
            counts['raw_person_detections']+=len(people);counts['frames_with_person']+=bool(people)
            counts['nonperson_detections']+=len(frame['instances'])-len(people)
            counts['assigned_observed_tracks']+=sum(t.get('global_id') is not None for t in frame['tracks'])
        run=json.loads((root/'run.json').read_text())
        result[name]=dict(path=str(root),counts=dict(counts),summary=summary,model=run['detector'],
            data_sha256=sha256((root/'data.json').read_bytes()).hexdigest(),
            scores=dict(min=min(scores,default=None),max=max(scores,default=None),
                mean=statistics.mean(scores) if scores else None,at_least_point5=sum(s>=.5 for s in scores),
                below_point4=sum(s<.4 for s in scores)))
    if not signatures or any(s!=signatures[0] for s in signatures[1:]):
        raise ValueError('Transfer runs do not use the same source observations')
    return dict(schema_version=1,runs=result,observations=len(signatures[0]),
        protocol='Matched timestamps, camera/epoch, frame index, source paths and dimensions. Whole-body GPU pose, OSNet and original cognitive tracker. No rotation recovery.',
        limitations='One unlabeled paired clip. Counts/assigned IDs are functional outputs, not detection or ReID accuracy. Cold run timing is not stable live capacity.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',required=True,help='Name=run-directory')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    runs={name:Path(path) for name,path in (spec.split('=',1) for spec in args.run)}
    if len(runs)!=len(args.run):raise ValueError('Duplicate transfer labels')
    result=summarize(runs);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Paired retail functional transfer check','',result['protocol'],'',result['limitations'],'',
        '| Model | Person detections | Frames with person | Assigned track observations | Max score | Scores >=0.5 | Nonperson detections |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in result['runs'].items():
        c=row['counts'];s=row['scores']
        lines.append(f'| {name} | {c["raw_person_detections"]} | {c["frames_with_person"]} | {c["assigned_observed_tracks"]} | {s["max"]:.4f} | {s["at_least_point5"]} | {c["nonperson_detections"]} |')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),observations=result['observations']),indent=2))


if __name__=='__main__':main()
