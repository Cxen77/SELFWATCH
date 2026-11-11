"""Evaluate explicitly complete, reviewed person identities across camera views.

ID scores use global bipartite matching of all threshold-compatible GT/predicted
identity pairs, consistent with TrackEval's Identity metric (not greedy matches):
https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py
Each camera observation is an evaluation timestep; subject names span cameras.
Only online recorded IDs are used by default. No automatic pose-derived labels.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracking import iou


def canonical(gid, aliases):
    seen=set()
    while str(gid) in aliases:
        if str(gid) in seen:
            raise ValueError('Cyclic identity aliases')
        seen.add(str(gid))
        gid=aliases[str(gid)]
    return str(gid)


def evaluate(data,annotations,threshold=.5,identity_view='online',allow_assisted=False,case=None,
             reference_source='manual'):
    if not 0<threshold<=1 or identity_view not in ('online','retrospective'):
        raise ValueError('Invalid evaluation policy')
    if data['run_id']!=annotations.get('run_id'):
        raise ValueError('Annotation run_id differs from prediction run')
    frames={f['observation_id']:f for f in data['frames']}
    if len(frames)!=len(data['frames']):
        raise ValueError('Duplicate prediction observation IDs')
    if case is not None and case not in data.get('tracking_cases',{}):
        raise ValueError('Unknown tracking case')
    identity=data.get('identity_by_case',{}).get(case,data.get('identity',{}))
    aliases={str(k):v for k,v in identity.get('aliases',{}).items()}
    gt_counts,pred_counts,potential=Counter(),Counter(),Counter()
    local_changes,global_changes=Counter(),Counter()
    last_match,by_camera={},defaultdict(Counter)
    traces,seen=[],set()
    reviewed=0
    selected=sorted(annotations['frames'],key=lambda a:(a['frame']['captured_at_s'],a['frame']['camera_id']))
    for annotation in selected:
        oid=annotation['observation_id']
        if oid in seen:
            raise ValueError('Duplicate annotation observation IDs')
        seen.add(oid)
        if oid not in frames:
            raise ValueError('Annotation refers to an unknown observation')
        if not annotation.get('reviewed') or 'person' not in annotation.get('exhaustive_labels',[]):
            continue
        source=frames[oid]
        for field in ('camera_id','stream_epoch','frame_index','captured_at_s','width','height','source'):
            if annotation['frame'].get(field)!=source['frame'].get(field):
                raise ValueError(f'Annotation source frame differs: {field}')
        marks=[m for m in annotation.get('marks',[]) if m.get('label')=='person' and m.get('kind')=='box']
        subjects=[]
        for mark in marks:
            if not allow_assisted and mark.get('source')!=reference_source:
                raise ValueError('Assisted/pseudo labels require explicit --allow-assisted; not independent ground truth')
            subject=mark.get('subject','').strip()
            if not subject:
                raise ValueError('Every visible person needs a persistent manual subject reference')
            box=mark.get('xyxy',[])
            if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('Invalid ground-truth person box')
            subjects.append(subject)
        if len(set(subjects))!=len(subjects):
            raise ValueError('A subject occurs twice in one camera observation')
        predictions=[]
        for track in source['tracks_by_case'][case] if case is not None else source.get('tracks',[]):
            if track.get('global_id') is None or track.get('status','observed')!='observed' or track.get('visual_suppressed'):
                continue
            index=track['instance_index']
            if not 0<=index<len(source['instances']) or source['instances'][index]['label'].lower()!='person':
                raise ValueError('Track refers to a non-person or absent detection')
            gid=str(track['global_id'])
            if identity_view=='retrospective':
                gid=canonical(gid,aliases)
            predictions.append((gid,source['instances'][index]['box_xyxy'],track['local_id']))
        pids=[p[0] for p in predictions]
        if len(set(pids))!=len(pids):
            raise ValueError('Two visible predictions own the same global ID in one camera observation')
        gt_counts.update(subjects)
        pred_counts.update(pids)
        overlaps=np.array([[iou(m['xyxy'],p[1]) for p in predictions] for m in marks],dtype=float).reshape(len(marks),len(predictions))
        for r,c in zip(*np.nonzero(overlaps>=threshold)):
            potential[subjects[r],pids[c]]+=1
        camera=source['frame']['camera_id']
        by_camera[camera].update(evaluated_frames=1,gt_detections=len(marks),predicted_detections=len(predictions))
        matched=[]
        if overlaps.size:
            # Diagnostic frame matching maximizes valid match count before IoU.
            score=(overlaps>=threshold).astype(float)+overlaps*1e-3
            rows,cols=linear_sum_assignment(-score)
            for r,c in zip(rows,cols):
                if overlaps[r,c]<threshold:
                    continue
                subject,gid=subjects[r],pids[c]
                key=(camera,source['frame']['stream_epoch'],subject)
                previous=last_match.get(key)
                if previous is not None:
                    global_changes[camera]+=previous[0]!=gid
                    local_changes[camera]+=previous[1]!=predictions[c][2]
                last_match[key]=(gid,predictions[c][2])
                matched.append(dict(subject=subject,global_id=gid,local_id=predictions[c][2],iou=float(overlaps[r,c])))
        by_camera[camera]['frame_box_matches']+=len(matched)
        traces.append(dict(observation_id=oid,camera_id=camera,captured_at_s=source['frame']['captured_at_s'],matches=matched))
        reviewed+=1
    if not reviewed:
        raise ValueError('No explicitly complete reviewed person frames; refusing an accuracy claim')
    gt_ids,pred_ids=sorted(gt_counts),sorted(pred_counts)
    matrix=np.array([[potential[g,p] for p in pred_ids] for g in gt_ids],dtype=np.int64).reshape(len(gt_ids),len(pred_ids))
    assignments=[]
    if matrix.size:
        rows,cols=linear_sum_assignment(-matrix)
        assignments=[dict(subject=gt_ids[r],global_id=pred_ids[c],compatible_detections=int(matrix[r,c]))
                     for r,c in zip(rows,cols) if matrix[r,c]>0]
    idtp=sum(a['compatible_detections'] for a in assignments)
    idfn=sum(gt_counts.values())-idtp
    idfp=sum(pred_counts.values())-idtp
    return dict(schema_version=1,run_id=data['run_id'],tracking_case=case,identity_view=identity_view,
        scope='Explicitly exhaustive reviewed camera observations only; no claim about unreviewed frames',
        threshold=threshold,allow_assisted=allow_assisted,reference_source=reference_source,
        evaluated_frames=reviewed,total_frames=len(frames),
        IDTP=idtp,IDFN=idfn,IDFP=idfp,IDF1=2*idtp/max(1,2*idtp+idfn+idfp),
        IDP=idtp/max(1,idtp+idfp),IDR=idtp/max(1,idtp+idfn),
        distinct_subjects=len(gt_ids),distinct_predicted_ids=len(pred_ids),
        sampled_global_id_changes=sum(global_changes.values()),sampled_local_id_changes=sum(local_changes.values()),
        sampled_change_semantics='Changes between matched reviewed observations per camera and stream epoch; not full-sequence CLEAR IDSW',
        per_camera={c:dict(v,sampled_global_id_changes=global_changes[c],sampled_local_id_changes=local_changes[c]) for c,v in by_camera.items()},
        global_assignment=assignments,frame_matches=traces,
        retrospective_warning='Final aliases intentionally use future information' if identity_view=='retrospective' else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Review run directory')
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iou',type=float,default=.5)
    parser.add_argument('--identity-view',choices=('online','retrospective'),default='online')
    parser.add_argument('--allow-assisted',action='store_true')
    parser.add_argument('--case',help='Case from a fixed-input study; annotations keep the same run_id')
    args=parser.parse_args()
    data_file=args.run/'data.json'
    result=evaluate(json.loads(data_file.read_text(encoding='utf-8')),
        json.loads(args.annotations.read_text(encoding='utf-8')),args.iou,args.identity_view,args.allow_assisted,args.case)
    result['inputs']=dict(predictions_sha256=sha256(data_file.read_bytes()).hexdigest(),
        annotations_sha256=sha256(args.annotations.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frame_matches','global_assignment')},indent=2))


if __name__=='__main__':
    main()
