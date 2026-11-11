"""Audited WEPDTOF person-box subset. No synthetic identity ground truth.

Published boxes are rotated. Our diagnostic uses their clipped polygons'
enclosing axis-aligned boxes at IoU .5, not the authors' rotated-box AP.
"""
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracking import iou


def contained(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Dataset path escaped root')
    return path


def ground_truth_polygon(box,width,height):
    if len(box)!=5 or not all(math.isfinite(x) for x in box) or min(box[2:4])<=0:
        raise ValueError('Invalid rotated person annotation')
    cx,cy,w,h,angle=box
    radians=math.radians(angle)
    matrix=np.array([[math.cos(radians),-math.sin(radians)],[math.sin(radians),math.cos(radians)]])
    points=np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])@matrix.T+[cx,cy]
    bounds=np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)
    area,clipped=cv2.intersectConvexConvex(points.astype(np.float32),bounds)
    if clipped is None or area<=1:raise ValueError('Annotation has no positive in-frame area')
    polygon=clipped.reshape(-1,2)
    return tuple(tuple(map(float,p)) for p in polygon)


def in_roi(box,mask,center=False):
    cx,cy=(box[0],box[1]) if center else ((box[0]+box[2])/2,(box[1]+box[3])/2)
    return (0<=cx<mask.shape[1] and 0<=cy<mask.shape[0] and mask[int(cy),int(cx)]>=128)


def load_selection(root):
    root=Path(root)
    selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'download_manifest.json').read_text(encoding='utf-8'))
    if selection['status']!='complete' or manifest['status']!='complete':
        raise ValueError('Incomplete dataset download')
    for file in manifest['files']:
        path=contained(root,file['path'])
        if path.stat().st_size!=file['size'] or sha256(path.read_bytes()).hexdigest()!=file['sha256']:
            raise ValueError('Dataset file checksum mismatch')
    # Freeze separate validation scenes and retain four final-test scene groups
    # before running models. No frame-level random splitting across one video.
    groups=selection['group_order'];n=len(groups)//2
    roles={group:'development' if i<n else 'validation' if i<n+3 else 'sealed_test'
           for i,group in enumerate(groups)}
    if len(groups)!=14:raise ValueError('Unexpected source scene groups; audit split before inference')
    return selection,roles


def load_image(root,item):
    image=cv2.imread(str(contained(root,item['image_path'])))
    mask=cv2.imread(str(contained(root,item['roi_path'])),cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2]!=mask.shape:
        raise ValueError('Image/ROI cannot be decoded or dimensions differ')
    if sha256(contained(root,item['image_path']).read_bytes()).hexdigest()!=item['image_sha256']:
        raise ValueError('Selected image checksum mismatch')
    height,width=image.shape[:2]
    positives=[];excluded=Counter()
    for raw in item['annotations']:
        if raw['category_id']!=1:raise ValueError('Unexpected annotation class')
        if raw.get('ignore') or raw.get('iscrowd'):raise ValueError('Ignore/crowd protocol needs a separate audit')
        if not in_roi(raw['bbox'],mask,center=True):
            excluded['ground_truth_outside_roi']+=1;continue
        polygon=ground_truth_polygon(raw['bbox'],width,height)
        box=(min(p[0] for p in polygon),min(p[1] for p in polygon),max(p[0] for p in polygon),max(p[1] for p in polygon))
        positives.append(dict(box_xyxy=box,polygon=polygon,person_id=raw.get('person_id'),source_bbox=raw['bbox']))
    # The same input mask is applied to EVERY candidate. Pixels outside the
    # declared ROI are neutral 114; raw images remain unchanged on disk.
    processed=image.copy();processed[mask<128]=114
    return processed,mask,positives,dict(excluded)


def match_boxes(ground_truth,predictions,threshold=.5):
    overlaps=np.array([[iou(g,p) for p in predictions] for g in ground_truth],dtype=float).reshape(len(ground_truth),len(predictions))
    matches=[]
    if overlaps.size:
        score=(overlaps>=threshold).astype(float)+overlaps*1e-3
        rows,cols=linear_sum_assignment(-score)
        matches=[dict(gt=int(r),prediction=int(c),iou=float(overlaps[r,c])) for r,c in zip(rows,cols) if overlaps[r,c]>=threshold]
    return dict(true_positive=len(matches),false_positive=len(predictions)-len(matches),
                false_negative=len(ground_truth)-len(matches),matches=matches)
