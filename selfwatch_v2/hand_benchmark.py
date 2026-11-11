"""Frozen hand-refinement/device ablation on cached person/whole-body outputs."""
import argparse
from collections import Counter,defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .wholebody_benchmark import load_dataset,file_hash
from .pose_metrics import score_person,finalize
from .hand_refinement import HandPose,hand_roi

PROTOCOL=dict(name='Hand refinement diagnostic v2',max_refined_hands_per_image=6,
    roi='At least 8 of 21 whole-body hand landmarks with raw score >=0.45; enclosing reliable points, minimum 16px side, clip to image. RTMLib applies its normal crop padding.',
    selection='Top 6 ROIs by median qualifying landmark score, then person index and left/right order. No GT used for live-style ROI construction or selection.',
    replacement='Replace all 21 points of each selected hand, preserve unselected baseline outputs; no GT or cross-model score cherry-picking.',
    reference='Additional hand GT-box case is diagnostic only, unavailable at deployment.',
    metric='Same reliable GT hand joints; PCK at 10% sqrt(original GT hand box area). Missing people/pose remain incorrect in the same denominator.',
    confidence='Raw landmark scores are not calibrated probabilities; also report gated PCK at score>=0.45.',
    scheduling='Explicit CPU/CUDA device and actual providers recorded; 2 intra-op threads; one ROI forward at a time. No live CPU/GPU scheduling integration yet.',
    limitation='Public validation diagnostic, single selected subset; not official AP, unseen retail performance or a novelty claim. No fine-tuning.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/coco-wholebody-pilot-v2'))
    parser.add_argument('--baseline',type=Path,default=Path('runs/wholebody-validation-v2'))
    parser.add_argument('--weights',type=Path,default=Path('weights/v2/hand/pose.onnx'))
    parser.add_argument('--hand-device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    data,manifest=load_dataset(args.dataset)
    baseline=json.loads((args.baseline/'results.json').read_text())
    if (baseline['input_lock']['selection_sha256']!=manifest['selection_sha256'] or baseline['source_changed_during_run']):
        raise ValueError('Cached baseline is not the frozen stable evaluation')
    with gzip.open(args.baseline/'predictions.jsonl.gz','rt',encoding='utf-8') as cache:records=[json.loads(line) for line in cache]
    if [r['image_id'] for r in records]!=[i['id'] for i in data['images']]:raise ValueError('Cached baseline image order differs')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'source').mkdir()
    sources={}
    for name in ('hand_benchmark.py','hand_refinement.py','pose_metrics.py'):
        path=Path(__file__).parent/name;payload=path.read_bytes();sources[name]=sha256(payload).hexdigest()
        (args.output/'source'/name).write_bytes(payload)
    lock=dict(protocol=PROTOCOL,selection_sha256=manifest['selection_sha256'],source_sha256=sources,
        baseline_results_sha256=file_hash(args.baseline/'results.json'),baseline_cache_sha256=file_hash(args.baseline/'predictions.jsonl.gz'),
        hand_model_sha256=file_hash(args.weights),hand_device=args.hand_device)
    (args.output/'input_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    if args.hand_device=='cuda':
        from .gpu_runtime import activate
        activate()
        import torch
        torch.set_num_threads(2)
    model=HandPose(args.weights,args.hand_device)
    model.predict(np.zeros((256,256,3),np.uint8),(20,20,200,200))
    cases=('wholebody_oracle','wholebody_pipeline','hand_oracle','roi_refined_pipeline')
    totals={case:defaultdict(Counter) for case in cases};counts=Counter();times=defaultdict(list);rows=[]
    annotation_map={a['id']:a for a in data['annotations']}
    with gzip.open(args.output/'refined.jsonl.gz','wt',encoding='utf-8') as output:
        for item,record in zip(data['images'],records):
            path=args.dataset/'images'/item['file_name'];image=cv2.imread(str(path))
            if image is None or sha256(image.tobytes()).hexdigest()!=record['processed_pixel_sha256']:
                raise ValueError('Refinement pixels differ from baseline')
            gt=[annotation_map[i] for i in record['ground_truth_ids']]
            oracle=[np.array(p['keypoints_xyc'],dtype=float) for p in record['oracle']['instances']]
            people=[p for p in record['detector']['instances'] if p['label']=='person']
            before=[np.array(p['keypoints_xyc'],dtype=float) if p['keypoints_xyc'] is not None else None for p in people]
            refined=[p.copy() if p is not None else None for p in before];candidates=[]
            tick=time.perf_counter()
            for index,points in enumerate(before):
                if points is None:continue
                for side,offset in (('left',91),('right',112)):
                    roi=hand_roi(points[offset:offset+21],item['width'],item['height'])
                    if roi is not None:candidates.append(dict(person=index,side=side,offset=offset,**roi))
            candidates.sort(key=lambda r:(-r['score'],r['person'],r['offset']))
            selected=candidates[:6]
            for roi in selected:refined[roi['person']][roi['offset']:roi['offset']+21]=model.predict(image,roi['box'])
            times['incremental_roi_ms'].append((time.perf_counter()-tick)*1000)
            counts.update(images=1,roi_candidates=len(candidates),refined_hands=len(selected),deferred_rois=max(0,len(candidates)-6))
            matched={m['gt']:m['prediction'] for m in record['matches']['matches']}
            per_image={case:defaultdict(Counter) for case in cases};specialized=[]
            tick=time.perf_counter()
            for index,annotation in enumerate(gt):
                specialized_oracle=oracle[index].copy()
                for side,offset in (('lefthand',91),('righthand',112)):
                    box=annotation[side+'_box'];points=np.array(annotation[side+'_kpts']).reshape(21,3)
                    if annotation[side+'_valid'] and np.any(points[:,2]>0) and min(box[2:])>0:
                        x,y,w,h=box
                        specialized_oracle[offset:offset+21]=model.predict(image,(x,y,x+w,y+h));counts['oracle_hands']+=1
                specialized.append(specialized_oracle.tolist())
                matched_index=matched.get(index)
                variants=dict(wholebody_oracle=oracle[index],hand_oracle=specialized_oracle,
                    wholebody_pipeline=before[matched_index] if matched_index is not None else None,
                    roi_refined_pipeline=refined[matched_index] if matched_index is not None else None)
                for case,prediction in variants.items():
                    metrics,_=score_person(annotation,prediction)
                    for name,stats in metrics.items():
                        if 'hand' in name:totals[case][name].update(stats);per_image[case][name].update(stats)
            times['oracle_hand_ms'].append((time.perf_counter()-tick)*1000)
            rows.append(dict(image_id=item['id'],metrics={case:finalize(stats) for case,stats in per_image.items()}))
            output.write(json.dumps(dict(image_id=item['id'],rois=selected,
                refined=[p.tolist() if p is not None else None for p in refined],oracle=specialized),allow_nan=False)+'\n')
            if counts['images']%8==0:print(f'{counts["images"]}/64 images hand-refined on {args.hand_device}',flush=True)
    metrics={case:finalize(stats) for case,stats in totals.items()}
    for case,reference in (('wholebody_oracle','oracle_crops'),('wholebody_pipeline','detector_pipeline')):
        for name,row in metrics[case].items():
            expected=baseline['metrics'][reference][name]
            if any(row[key]!=expected[key] for key in ('eligible','correct','confident','gated_correct')):
                raise ValueError('Cached baseline hand metrics failed exact reconciliation')
    result=dict(protocol=PROTOCOL,input_lock=lock,counts=dict(counts),metrics=metrics,images=rows,
        model=model.provenance,timing={key:dict(mean=float(np.mean(values)),p95=sorted(values)[math.ceil(.95*len(values))-1]) for key,values in times.items()},
        source_changed_during_run=any(file_hash(Path(__file__).parent/name)!=digest for name,digest in sources.items()))
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Hand-refinement diagnostic: '+args.hand_device,'',PROTOCOL['limitation'],'',
        '| Case | Part | GT joints | Raw PCK | Gated PCK | Confident coverage |','|---|---|---:|---:|---:|---:|']
    for case,parts in metrics.items():
        for name,row in parts.items():lines.append(f'| {case} | {name} | {row["eligible"]} | {row["PCK"]:.3f} | {row["gated_PCK"]:.3f} | {row["confidence_coverage"]:.3f} |')
    lines+=['',PROTOCOL['roi'],'',PROTOCOL['selection'],'',PROTOCOL['replacement'],'',PROTOCOL['scheduling'],
        '', 'Hand GT-box inference is an oracle diagnostic. The predicted-ROI experiment uses no GT to construct or select crops. Unselected or unusable hands retain baseline predictions.',
        '', 'Incremental hand-model timing excludes the cached GPU detector/whole-body work and ReID; it is not live three-camera FPS.',
        '', 'Timing: `'+json.dumps(result['timing'])+'`']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts=result['counts'],hands={case:p['hands'] for case,p in metrics.items()},timing=result['timing']),indent=2))


if __name__=='__main__':main()
