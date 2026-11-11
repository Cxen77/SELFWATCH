"""Locate where synchronized appearance evidence is lost, using frozen outputs."""
import argparse
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .identity_study import digest
from .overhead_data import match_boxes


def analyse(root):
    result=json.loads((root/'results.json').read_text())
    for file,key in (('predictions.json','predictions_sha256'),('annotations.json','annotations_sha256'),
                     ('features.npz','features_sha256')):
        if digest(root/file)!=result[key]:
            raise ValueError('Tracking evidence checksum mismatch')
    frames=json.loads((root/'predictions.json').read_text())['frames']
    labels={row['observation_id']:row['marks'] for row in json.loads((root/'annotations.json').read_text())['frames']}
    synchronized=defaultdict(list)
    with np.load(root/'features.npz',allow_pickle=False) as vectors:
        for frame in frames:
            marks=labels[frame['observation_id']]
            indices=[i for i,p in enumerate(frame['instances']) if p['label']=='person']
            matches=match_boxes([m['xyxy'] for m in marks],[frame['instances'][i]['box_xyxy'] for i in indices],.5)
            features={int(i):vectors[key] for i,key in frame['feature_keys'].items()}
            subjects={marks[m['gt']]['subject']:indices[m['prediction']] for m in matches['matches']}
            synchronized[frame['frame']['captured_at_s']].append(
                (frame['frame']['camera_id'],{m['subject'] for m in marks},subjects,features))
    totals=defaultdict(Counter)
    threshold=.78;margin=.06
    for group in synchronized.values():
        for (ca,ga,da,fa),(cb,gb,db,fb) in combinations(group,2):
            row=totals[f'{ca}/{cb}']
            row['labelled_same_person_pairs']+=len(ga&gb)
            for subject in da.keys() & db.keys():
                row['both_detected']+=1
                ia,ib=da[subject],db[subject]
                if ia not in fa or ib not in fb:
                    continue
                row['both_have_eligible_embeddings']+=1
                cosine=float(np.dot(fa[ia],fb[ib]))
                if cosine<threshold:
                    continue
                row['above_cosine_threshold']+=1
                alt_a=max((float(np.dot(fa[ia],v)) for i,v in fb.items() if i!=ib),default=-1.)
                alt_b=max((float(np.dot(fb[ib],v)) for i,v in fa.items() if i!=ia),default=-1.)
                row['passes_both_margins']+=int(cosine-max(alt_a,alt_b)>=margin)
    combined=Counter()
    for row in totals.values():
        combined.update(row)
    return dict(scope='Per-timestamp crop diagnostic with published projected boxes. It does not replay gallery confirmation or measure live association.',
        cosine_threshold=threshold,margin=margin,totals=dict(combined),per_pair={k:dict(v) for k,v in totals.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    result=analyse(args.run)
    with (args.run/'appearance-funnel.json').open('x',encoding='utf-8') as target:
        json.dump(result,target,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
