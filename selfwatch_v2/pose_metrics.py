"""Explicit part-normalized PCK diagnostics; these are not official OKS AP."""
from collections import Counter
import math
import numpy as np

PARTS=(('body','keypoints',0,17,None),('foot','foot_kpts',17,6,'foot_valid'),
       ('face','face_kpts',23,68,'face_valid'),('left_hand','lefthand_kpts',91,21,'lefthand_valid'),
       ('right_hand','righthand_kpts',112,21,'righthand_valid'))


def person_box(annotation,width,height):
    box=annotation['bbox']
    if len(box)!=4 or not all(math.isfinite(x) for x in box) or min(box[2:])<=0:
        raise ValueError('Invalid COCO person box')
    x,y,w,h=box;result=(max(0,x),max(0,y),min(width,x+w),min(height,y+h))
    if result[2]<=result[0] or result[3]<=result[1]:raise ValueError('Person box has no in-frame area')
    return result


def targets(annotation):
    result=[]
    for name,key,offset,size,valid in PARTS:
        values=np.asarray(annotation[key],dtype=float)
        if values.size!=size*3 or not np.isfinite(values).all():raise ValueError('Invalid whole-body landmark layout')
        points=values.reshape(size,3)
        if np.any(points[:,2]<0):raise ValueError('Negative annotation reliability')
        if name=='body':
            if not np.isin(points[:,2],[0,1,2]).all():raise ValueError('Body visibility must use COCO 0/1/2 flags')
            for group,mask in (('body_labeled',points[:,2]>0),('body_visible',points[:,2]==2),('body_occluded',points[:,2]==1)):
                result.append((group,points,offset,mask,annotation['bbox'],.05))
        elif bool(annotation[valid]):
            box=annotation['bbox'] if name=='foot' else annotation[{'face':'face_box','left_hand':'lefthand_box','right_hand':'righthand_box'}[name]]
            result.append((name,points,offset,points[:,2]>0,box,.05 if name=='foot' else .10))
    return result


def score_person(annotation,prediction,score_threshold=.45):
    if prediction is not None:
        prediction=np.asarray(prediction,dtype=float)
        if prediction.shape!=(133,3) or not np.isfinite(prediction).all():raise ValueError('Predictions must have 133 finite x/y/score triples')
    result={};excluded=Counter()
    for name,points,offset,eligible,box,alpha in targets(annotation):
        if not eligible.any():continue
        if len(box)!=4 or not all(math.isfinite(x) for x in box) or min(box[2:])<=0:
            excluded['invalid_part_box']+=1;continue
        scale=math.sqrt(box[2]*box[3]);count=int(eligible.sum())
        stats=Counter(eligible=count,available=0,confident=0,correct=0,gated_correct=0,error_sum=0.)
        if prediction is not None:
            selected=prediction[offset:offset+len(points)][eligible]
            errors=np.linalg.norm(selected[:,:2]-points[eligible,:2],axis=1)/scale
            confident=selected[:,2]>=score_threshold;correct=errors<=alpha
            stats.update(available=count,confident=int(confident.sum()),correct=int(correct.sum()),
                gated_correct=int((confident&correct).sum()),error_sum=float(errors.sum()))
        result[name]=stats
        if name in ('left_hand','right_hand'):
            result.setdefault('hands',Counter()).update(stats)
            side=min(box[2:]);bucket='hands_lt16px' if side<16 else 'hands_16to32px' if side<32 else 'hands_ge32px'
            result.setdefault(bucket,Counter()).update(stats)
    return result,excluded


def finalize(counters):
    result={}
    for name,values in counters.items():
        row=dict(values);n=values['eligible'];available=values['available'];confident=values['confident']
        row.update(PCK=values['correct']/n if n else None,
            gated_PCK=values['gated_correct']/n if n else None,
            confidence_coverage=confident/n if n else None,
            pose_availability=available/n if n else None,
            PCK_given_confident=values['gated_correct']/confident if confident else None,
            conditional_mean_normalized_error=values['error_sum']/available if available else None)
        result[name]=row
    return result
