"""Independent person annotations, scene-separated development/validation.

Final-test scene groups are sealed. This pilot reports axis-aligned diagnostic
precision/recall at confidence .25 and IoU .5, not published rotated AP or IDF1.
"""
import argparse
from collections import Counter
import copy
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import time

import cv2
from .contracts import FrameRef
from .overhead_data import load_selection,load_image,in_roi,match_boxes
from .rotation import RotationPersonPerception
from .yolo import YoloPerception


CASES={'native_640':'YOLO26n-seg / 640','native_960':'YOLO26n-seg / 960',
       'rotate_640':'Four orientations / 640','selective_640':'Weak-view recovery / 640'}


def summarize(rows):
    total=Counter()
    for row in rows:total.update({k:row[k] for k in ('true_positive','false_positive','false_negative')})
    tp,fp,fn=(total[k] for k in ('true_positive','false_positive','false_negative'))
    times=sorted(r['latency_ms'] for r in rows)
    return dict(**total,images=len(rows),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
        F1=2*tp/max(1,2*tp+fp+fn),mean_ms=statistics.mean(times) if times else None,
        p95_ms=times[max(0,math.ceil(.95*len(times))-1)] if times else None,
        mean_passes=statistics.mean(r['passes'] for r in rows) if rows else None)


def benchmark(args):
    import torch
    import numpy as np
    torch.set_num_threads(2)
    selected,roles=load_selection(args.dataset)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'images').mkdir()
    source_files=[Path(__file__).parent/name for name in
        ('overhead_benchmark.py','overhead_data.py','rotation.py','yolo.py','contracts.py','tracking.py')]
    (args.output/'source').mkdir()
    source_hashes={}
    for path in source_files:
        payload=path.read_bytes();source_hashes[path.name]=sha256(payload).hexdigest()
        (args.output/'source'/path.name).write_bytes(payload)
    split_lock=dict(roles=roles,selection_sha256=sha256((args.dataset/'selection.json').read_bytes()).hexdigest(),
        final_test_policy='Sealed: no final-test image is decoded or inferred in this run')
    (args.output/'split_lock.json').write_text(json.dumps(split_lock,indent=2),encoding='utf-8')
    base=YoloPerception(args.weights,device=args.device,imgsz=640,confidence=.25)
    case_labels=dict(CASES)
    case_labels.update(native_640=f'{base.model.task} checkpoint {args.weights.name} / 640',
                       native_960=f'{base.model.task} checkpoint {args.weights.name} / 960')
    large=copy.copy(base);large.imgsz=960
    # Shared weighted model/backend; GPU calls remain sequential.
    models={'native_640':base,'native_960':large,
            'rotate_640':RotationPersonPerception(base,mode='always'),
            'selective_640':RotationPersonPerception(base,mode='weak')}
    warm_image=np.zeros((640,640,3),dtype=np.uint8)
    for name,model in models.items():
        f=FrameRef('warmup',name,0,0.,time.time(),640,640,'synthetic_warmup','independent_image')
        model.predict(f,warm_image)
    if args.device!='cpu':torch.cuda.reset_peak_memory_stats()
    rows={role:{name:[] for name in CASES} for role in ('development','validation')}
    records=[];choice=None
    started=time.perf_counter()
    with gzip.open(args.output/'observations.jsonl.gz','wt',encoding='utf-8') as cache:
        for role in ('development','validation'):
            subset=[r for r in selected['items'] if roles[r['source_group']]==role]
            if args.max_per_partition:subset=subset[:args.max_per_partition]
            for position,item in enumerate(subset):
                image,roi,gt,excluded=load_image(args.dataset,item)
                height,width=image.shape[:2]
                frame=FrameRef(item['sequence'],f'independent-image:{item["image_id"]}',0,0.,time.time(),
                    width,height,str((args.dataset/item['image_path']).resolve()),'independent_image_no_video_time')
                record=dict(image_id=item['image_id'],sequence=item['sequence'],source_group=item['source_group'],
                    partition=role,width=width,height=height,ground_truth=gt,excluded=excluded,cases={},
                    raw_source_sha256=item['image_sha256'],processed_pixel_sha256=sha256(image.tobytes()).hexdigest(),
                    image=f'images/{item["image_id"]}.png')
                # Rotate case order to reduce systematic warm-cache preference.
                names=list(CASES);names=names[position%len(names):]+names[:position%len(names)]
                for name in names:
                    tick=time.perf_counter();observation,detail=models[name].predict(frame,image)
                    elapsed=(time.perf_counter()-tick)*1000
                    people=[p for p in observation.instances if p.label.lower()=='person']
                    predictions=[dict(box_xyxy=p.box_xyxy,score=p.score) for p in people if in_roi(p.box_xyxy,roi)]
                    matches=match_boxes([p['box_xyxy'] for p in gt],[p['box_xyxy'] for p in predictions])
                    passes=len(detail['passes']) if name in ('rotate_640','selective_640') else 1
                    result=dict(predictions=predictions,**matches,latency_ms=elapsed,passes=passes,
                                excluded_predictions=len(people)-len(predictions))
                    rows[role][name].append(result);record['cases'][name]=result
                    cache.write(json.dumps(dict(case=name,partition=role,image_id=item['image_id'],
                        observation=observation.to_dict()),allow_nan=False)+'\n')
                if not cv2.imwrite(str(args.output/record['image']),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                    raise RuntimeError('Cannot write lossless benchmark image')
                records.append(record)
                if (position+1)%8==0:print(f'{role}: {position+1}/{len(subset)} images',flush=True)
            if role=='development':
                stats={name:summarize(values) for name,values in rows[role].items()}
                quality=max(stats,key=lambda name:(stats[name]['F1'],-stats[name]['p95_ms']))
                eligible=[name for name in stats if stats[name]['p95_ms']<=args.budget_ms]
                budget=max(eligible,key=lambda name:(stats[name]['F1'],-stats[name]['p95_ms'])) if eligible else min(stats,key=lambda name:stats[name]['p95_ms'])
                choice=dict(quality_winner=quality,budget_winner=budget,detector_budget_ms=args.budget_ms,
                    budget_satisfied=bool(eligible),selection_source='development only; frozen before validation',
                    development=stats)
                (args.output/'candidate_selection.json').write_text(json.dumps(choice,indent=2),encoding='utf-8')
                print('Frozen development selection: '+json.dumps(choice),flush=True)
    metrics={role:{name:summarize(values) for name,values in cases.items()} for role,cases in rows.items()}
    report=dict(schema_version=1,name='SELFWATCH overhead detector validation',cases=case_labels,
        dataset='WEPDTOF',dataset_source=selected['source'],method=__doc__,split_lock=split_lock,
        settings=vars(args),model=base.provenance,rotation=models['rotate_640'].provenance,
        metrics=metrics,choice=choice,frames=records,wall_seconds=time.perf_counter()-started,
        source_sha256=source_hashes,
        source_changed_during_run=any(sha256(path.read_bytes()).hexdigest()!=source_hashes[path.name] for path in source_files),
        hardware=dict(device=args.device,gpu_name=torch.cuda.get_device_name() if args.device!='cpu' else None,
            torch_threads=torch.get_num_threads(),torch_version=torch.__version__),
        memory=dict(torch_peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2 if args.device!='cpu' else 0,
            note='This process Torch allocator only; excludes other GPU processes and drivers'),
        protocol=dict(confidence=.25,iou=.5,input='ROI masked to neutral 114 for every candidate',
            roi_rule='Nonzero-valid source ROI; center pixels >=128; both GT and predicted centers must lie inside',
            box_rule='Original rotated annotations clipped to image; enclosing axis-aligned target boxes',
            timing='Detector-only latency after warmup, including preprocessing/mapping/NMS; excludes decoding, pose, ReID and export',
            latency_quantile='Nearest-rank P95; v1 archived runner used lower empirical percentile',
            exclusions='Final-test scenes remain sealed; no tracking, pose, item or payment accuracy measured'))
    (args.output/'data.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False),encoding='utf-8')
    lines=['# Overhead person-detection validation','',
        'Independent WEPDTOF annotations; bounded scene-separated pilot. This is not the authors\' rotated-box AP benchmark.','',
        '| Partition | Candidate | TP | FP | FN | Precision | Recall | F1 | Mean ms | P95 ms | Passes |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for role,models_stats in metrics.items():
        for name,s in models_stats.items():
            lines.append(f'| {role} | {case_labels[name]} | {s["true_positive"]} | {s["false_positive"]} | {s["false_negative"]} | {s["precision"]:.3f} | {s["recall"]:.3f} | {s["F1"]:.3f} | {s["mean_ms"]:.1f} | {s["p95_ms"]:.1f} | {s["mean_passes"]:.2f} |')
    lines.extend(['',f'Development quality winner: {choice["quality_winner"]}. Budget candidate: {choice["budget_winner"]}.',
        '', 'Final-test groups were not decoded or inferred. This does not establish retail deployment accuracy or live multi-camera FPS.'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(metrics=metrics,choice=choice,memory=report['memory']),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/wepdtof-pilot'))
    parser.add_argument('--weights',type=Path,default=Path('weights/v2/yolo26n-seg.pt'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='0')
    parser.add_argument('--budget-ms',type=float,default=50.)
    parser.add_argument('--max-per-partition',type=int)
    args=parser.parse_args()
    if args.budget_ms<=0 or (args.max_per_partition is not None and args.max_per_partition<1):raise ValueError('Positive limits required')
    benchmark(args)


if __name__=='__main__':main()
