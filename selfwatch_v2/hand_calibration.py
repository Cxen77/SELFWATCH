"""Select an operating point on development labels; evaluate it on disjoint images."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .pose_metrics import targets
from .wholebody_benchmark import file_hash, load_dataset

PRECISION=.75
COVERAGE=.10
GRID=tuple(i/1000 for i in range(0,1001,25))


def score(scores, correct, eligible, threshold):
    mask=np.zeros(len(scores),dtype=bool) if threshold is None else scores>=threshold
    emitted=int(mask.sum())
    hits=int((mask & correct).sum())
    return dict(threshold=threshold,eligible=eligible,available=len(scores),emitted=emitted,
        correct=hits,precision=hits/emitted if emitted else None,
        candidate_coverage=emitted/len(scores) if len(scores) else 0.,
        all_joint_coverage=emitted/eligible,correct_yield=hits/eligible)


def passes(row):
    return (row['precision'] or 0.)>=PRECISION and row['candidate_coverage']>=COVERAGE


def choose_threshold(scores,correct,eligible):
    candidates=[score(scores,correct,eligible,t) for t in GRID]
    feasible=[r for r in candidates if passes(r)]
    chosen=max(feasible,key=lambda r:(r['correct'],r['precision'],r['threshold'])) if feasible else score(scores,correct,eligible,None)
    return chosen,candidates


def read_cache(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def gather(root,baseline,hand,partition):
    data,manifest=load_dataset(root)
    if data.get('partition')!=partition or manifest.get('partition')!=partition:
        raise ValueError('Calibration partition mismatch')
    base_path=baseline/'predictions.jsonl.gz'
    hand_path=hand/'refined.jsonl.gz'
    report=json.loads((hand/'results.json').read_text())
    lock=report['input_lock']
    if (report['source_changed_during_run'] or lock['selection_sha256']!=manifest['selection_sha256']
        or lock['baseline_cache_sha256']!=file_hash(base_path)):
        raise ValueError('Predictions do not match frozen data and baseline')
    base=read_cache(base_path); refined=read_cache(hand_path)
    ids=[item['id'] for item in data['images']]
    if any([row['image_id'] for row in rows]!=ids for rows in (base,refined)):
        raise ValueError('Missing or reordered predictions')
    annotations={a['id']:a for a in data['annotations']}
    scores=[];correct=[];eligible=0
    for before,after in zip(base,refined):
        matches={m['gt']:m['prediction'] for m in before['matches']['matches']}
        selected={(roi['person'],roi['offset']) for roi in after['rois']}
        for index,annotation_id in enumerate(before['ground_truth_ids']):
            annotation=annotations[annotation_id]
            if annotation['image_id']!=before['image_id']:
                raise ValueError('Ground truth belongs to another image')
            person=matches.get(index)
            for name,points,offset,mask,box,alpha in targets(annotation):
                if name not in ('left_hand','right_hand') or min(box[2:])<=0:
                    continue
                eligible+=int(mask.sum())
                if (person,offset) not in selected:
                    continue
                pred=np.asarray(after['refined'][person])[offset:offset+21][mask]
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite specialist predictions')
                scores.extend(pred[:,2])
                correct.extend(np.linalg.norm(pred[:,:2]-points[mask,:2],axis=1)<=alpha*math.sqrt(box[2]*box[3]))
    if not eligible or not scores:
        raise ValueError('No labelled specialist candidates to evaluate')
    provenance=dict(selection_sha256=manifest['selection_sha256'],annotation_sha256=data['annotation_sha256'],
        image_ids=ids,image_hashes=[item['sha256'] for item in manifest['files']],
        hand_model_sha256=lock['hand_model_sha256'],baseline_results_sha256=lock['baseline_results_sha256'],
        refinement_protocol=report['protocol'],hand_results_sha256=file_hash(hand/'results.json'),
        hand_predictions_sha256=file_hash(hand_path))
    return np.asarray(scores),np.asarray(correct,dtype=bool),eligible,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('fit','evaluate'))
    for name in ('dataset','baseline','hand','policy','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    partition='development' if args.mode=='fit' else 'evaluation'
    scores,correct,eligible,provenance=gather(args.dataset,args.baseline,args.hand,partition)
    if args.mode=='fit':
        chosen,grid=choose_threshold(scores,correct,eligible)
        policy=dict(schema_version=1,source=provenance,operating_point=chosen,
            minimum_precision=PRECISION,minimum_candidate_coverage=COVERAGE,
            method='Fixed grid; maximize correct emitted joints subject to development precision/coverage floors. Abstain if no point qualifies.')
        with args.policy.open('x',encoding='utf-8') as target:
            json.dump(policy,target,indent=2)
        result=dict(policy=policy,development_grid=grid)
    else:
        policy=json.loads(args.policy.read_text())
        fit=policy['source']
        for key in ('image_ids','image_hashes'):
            if set(fit[key]) & set(provenance[key]):
                raise ValueError('Evaluation images overlap development')
        for key in ('annotation_sha256','hand_model_sha256','refinement_protocol'):
            if fit[key]!=provenance[key]:
                raise ValueError(f'Evaluation differs from development: {key}')
        frozen=score(scores,correct,eligible,policy['operating_point']['threshold'])
        result=dict(policy_sha256=file_hash(args.policy),evaluation=provenance,
            frozen=frozen,passed=passes(frozen),
            ungated=score(scores,correct,eligible,float(np.min(scores))),
            old_threshold=score(scores,correct,eligible,.45),
            scope='Reliable annotated joints of matched people; unqueried and missed joints count against all-joint yield. Public images, not retail transfer or overall hand visibility precision.')
    with args.output.open('x',encoding='utf-8') as target:
        json.dump(result,target,indent=2)
    print(json.dumps(result.get('frozen',policy['operating_point']),indent=2))


if __name__=='__main__':
    main()
