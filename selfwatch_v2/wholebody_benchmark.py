"""Frozen public validation PCK diagnostic, with oracle crops and detector misses."""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .contracts import FrameRef,Instance,Observation
from .overhead_data import contained,match_boxes
from .pose_metrics import score_person,finalize,person_box

PROTOCOL=dict(name='SELFWATCH COCO-WholeBody diagnostic v1.1',detector_confidence=.25,match_iou=.5,
    landmark_score_threshold=.45,max_detector_people_posed=6,pose_call_group_limit=3,actual_model_batch=1,
    body_PCK='distance <= 0.05 * sqrt(original annotated person bbox area)',
    foot_PCK='distance <= 0.05 * sqrt(original annotated person bbox area)',
    face_hand_PCK='distance <= 0.10 * sqrt(original annotated part bbox area)',
    body_labels='Separate v=2 visible, v=1 occluded and v>0 labeled metrics',
    part_labels='Only valid parts with reliability v>0; fractional reliability is NOT assumed to mean visible',
    confidence='Raw RTMW scores >=0.45; coverage and accuracy reported separately, not calibrated probabilities',
    misses='Missing detector association or unposed match contributes zero correct joints, with unchanged GT denominator',
    comparison='Oracle: all noncrowd GT person boxes. Detector: YOLO26n-seg at 640, pose top 6 people. Wrapper call groups <=3 ROIs; RTMLib executes one model forward per ROI (batch 1).',
    warning='Diagnostic PCK, not official OKS AP; public validation, not a new unseen retail test set. No model fitting or threshold tuning.')


def file_hash(path):
    digest=sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def load_dataset(root):
    manifest=json.loads((root/'download_manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Whole-body dataset download is incomplete')
    if file_hash(root/'selection.json')!=manifest['selection_sha256']:raise ValueError('Frozen selection changed')
    for record in manifest['files']:
        path=contained(root,record['path'])
        if path.stat().st_size!=record['bytes'] or file_hash(path)!=record['sha256']:
            raise ValueError('Dataset checksum mismatch')
    data=json.loads((root/'selection.json').read_text())
    if data['role']!='public_validation_diagnostic_only':raise ValueError('Unexpected dataset role')
    if len({i['id'] for i in data['images']})!=len(data['images']):raise ValueError('Duplicate validation image')
    return data,manifest


def refine_indices(pose,observation,image,indices):
    instances=list(observation.instances)
    for start in range(0,len(indices),3):
        chosen=indices[start:start+3]
        batch=replace(observation,instances=tuple(instances[i] for i in chosen))
        posed=pose.refine(batch,image)
        for i,instance in zip(chosen,posed.instances):instances[i]=instance
    return replace(observation,instances=tuple(instances),keypoint_schema='coco-wholebody133')


def benchmark(args):
    data,manifest=load_dataset(args.dataset)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'source').mkdir()
    sources={}
    for name in ('wholebody_benchmark.py','pose_metrics.py','wholebody.py','yolo.py','contracts.py','overhead_data.py','tracking.py'):
        path=Path(__file__).parent/name;payload=path.read_bytes();sources[name]=sha256(payload).hexdigest()
        (args.output/'source'/name).write_bytes(payload)
    lock=dict(protocol=PROTOCOL,selection_sha256=manifest['selection_sha256'],
        source_sha256=sources,settings=vars(args),annotation_sha256=data['annotation_sha256'])
    (args.output/'input_lock.json').write_text(json.dumps(lock,indent=2,default=str),encoding='utf-8')
    import torch
    torch.set_num_threads(2)
    if args.pose_device=='cuda':
        from .gpu_runtime import activate
        activate()
    from .wholebody import WholeBody
    from .yolo import YoloPerception
    detector=YoloPerception(args.weights,args.device,640,.25);detector.include_masks=False
    pose=WholeBody(args.pose_weights,args.pose_device,max_people=3)
    warm_image=np.zeros((640,640,3),np.uint8)
    warm_frame=FrameRef('warmup','synthetic',0,0.,time.time(),640,640,'synthetic','synthetic_warmup')
    detector.predict(warm_frame,warm_image)
    warm_observation=Observation(warm_frame,tuple(Instance((100,50,300,600),1.,0,'person') for _ in range(3)),'synthetic','detect',0.)
    pose.refine(warm_observation,warm_image)
    annotations=defaultdict(list)
    for annotation in data['annotations']:annotations[annotation['image_id']].append(annotation)
    totals={case:defaultdict(Counter) for case in ('oracle_crops','detector_pipeline')}
    counts=Counter();latencies=defaultdict(list);rows=[]
    started=time.perf_counter()
    with gzip.open(args.output/'predictions.jsonl.gz','wt',encoding='utf-8') as cache:
        for position,item in enumerate(data['images']):
            path=contained(args.dataset,'images/'+item['file_name']);image=cv2.imread(str(path))
            if image is None or image.shape[:2]!=(item['height'],item['width']):raise ValueError('Image dimensions differ from annotation')
            height,width=image.shape[:2];gt=[];boxes=[]
            for annotation in annotations[item['id']]:
                if annotation['category_id']!=1:raise ValueError('Unexpected whole-body annotation category')
                if annotation.get('iscrowd'):counts['crowd_annotations_excluded']+=1;continue
                try:box=person_box(annotation,width,height)
                except ValueError:counts['invalid_person_boxes_excluded']+=1;continue
                gt.append(annotation);boxes.append(box)
            frame=FrameRef('coco-val','independent-frozen-pilot',item['id'],0.,time.time(),width,height,str(path.resolve()),'independent_image_no_video_time')
            tick=time.perf_counter();raw,_=detector.predict(frame,image)
            indices=sorted((i for i,p in enumerate(raw.instances) if p.label=='person'),key=lambda i:raw.instances[i].score,reverse=True)[:6]
            predicted=refine_indices(pose,raw,image,indices)
            latencies['detector_pipeline_ms'].append((time.perf_counter()-tick)*1000)
            people=[p for p in predicted.instances if p.label=='person']
            matches=match_boxes(boxes,[p.box_xyxy for p in people],.5)
            matched={m['gt']:m['prediction'] for m in matches['matches']}
            counts.update(images=1,noncrowd_people=len(gt),detector_people=len(people),
                matched_people=len(matched),unmatched_predictions=matches['false_positive'],
                missed_people=matches['false_negative'],matched_without_pose=sum(people[p].keypoints_xyc is None for p in matched.values()))
            tick=time.perf_counter()
            oracle_input=Observation(frame,tuple(Instance(box,1.,0,'person') for box in boxes),'ground_truth_crops','detect',0.)
            oracle=refine_indices(pose,oracle_input,image,list(range(len(boxes))))
            latencies['oracle_crops_ms'].append((time.perf_counter()-tick)*1000)
            per_image={case:defaultdict(Counter) for case in totals}
            for index,annotation in enumerate(gt):
                candidates=dict(oracle_crops=oracle.instances[index].keypoints_xyc,
                    detector_pipeline=people[matched[index]].keypoints_xyc if index in matched else None)
                for case,points in candidates.items():
                    metrics,excluded=score_person(annotation,points,.45)
                    for name,stats in metrics.items():totals[case][name].update(stats);per_image[case][name].update(stats)
                    counts.update({case+'_'+key:value for key,value in excluded.items()})
            rows.append(dict(image_id=item['id'],people=len(gt),matches=len(matched),
                metrics={case:finalize(stats) for case,stats in per_image.items()}))
            cache.write(json.dumps(dict(image_id=item['id'],ground_truth_ids=[a['id'] for a in gt],
                image_sha256=file_hash(path),processed_pixel_sha256=sha256(image.tobytes()).hexdigest(),
                oracle=oracle.to_dict(),detector=predicted.to_dict(),matches=matches),allow_nan=False)+'\n')
            if (position+1)%8==0:print(f'{position+1}/{len(data["images"])} images evaluated',flush=True)
    report=dict(protocol=PROTOCOL,input_lock=lock,counts=dict(counts),
        metrics={case:finalize(stats) for case,stats in totals.items()},images=rows,
        timing={key:dict(mean=float(np.mean(values)),p95=sorted(values)[math.ceil(.95*len(values))-1]) for key,values in latencies.items()},
        timing_scope='Per-image diagnostic after synthetic warmup; includes only indicated inference/crop calls, not disk, labels, metrics, tracking or ReID. No live FPS claim.',
        detector=detector.provenance,pose=pose.provenance,source_sha256=sources,
        source_changed_during_run=any(file_hash(Path(__file__).parent/name)!=digest for name,digest in sources.items()),
        wall_seconds=time.perf_counter()-started)
    (args.output/'results.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False),encoding='utf-8')
    lines=['# Whole-body diagnostic on public COCO validation','',PROTOCOL['warning'],'',
        PROTOCOL['body_labels']+'. '+PROTOCOL['part_labels']+'.','',
        '| Case | Part | GT joints | Pose availability | Raw PCK | Confident coverage | Gated PCK |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case,metrics in report['metrics'].items():
        for name,row in metrics.items():
            lines.append(f'| {case} | {name} | {row["eligible"]} | {row["pose_availability"]:.3f} | {row["PCK"]:.3f} | {row["confidence_coverage"]:.3f} | {row["gated_PCK"]:.3f} |')
    lines+=['','Body/foot tolerance is 5% of sqrt(person bbox area); hand/face is 10% of sqrt(part bbox area). Gated PCK requires both correct location and raw landmark score >=0.45.',
        '',PROTOCOL['comparison'],
        '', 'Missing detections and crop-budget exclusions keep their GT denominator. Unmatched person boxes can include crowd/otherwise unscored people, so their count is not official detector FP/AP.',
        '',report['timing_scope'],'', 'Counts: `'+json.dumps(report['counts'])+'`']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts=report['counts'],metrics=report['metrics'],timing=report['timing']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/coco-wholebody-pilot-v2'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('weights/v2/yolo26n-seg.pt'))
    parser.add_argument('--pose-weights',type=Path,default=Path('weights/v2/rtmw/wholebody133.onnx'))
    parser.add_argument('--device',default='0');parser.add_argument('--pose-device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args();benchmark(args)


if __name__=='__main__':main()
