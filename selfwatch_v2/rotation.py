"""Optional person-only cardinal-rotation recovery using the same YOLO weights.

No guessed anatomy or projected people are added. Extra instances must be actual
model detections, mapped back before NMS, tracking, pose and durable storage.
"""
from dataclasses import replace
from hashlib import sha256
import json
import math
import time

from .tracking import iou


def inverse_point(point,width,height,angle,edges=False):
    x,y=point[:2]
    w,h=(width,height) if edges else (width-1,height-1)
    if angle==0:result=(x,y)
    elif angle==90:result=(y,h-x)
    elif angle==180:result=(w-x,h-y)
    elif angle==270:result=(w-y,x)
    else:raise ValueError('Only cardinal rotations are supported')
    return (*map(float,result),*point[2:])


def inverse_instance(person,frame,angle):
    x1,y1,x2,y2=person.box_xyxy
    corners=[inverse_point(p,frame.width,frame.height,angle,edges=True)
             for p in ((x1,y1),(x2,y1),(x2,y2),(x1,y2))]
    box=(max(0.,min(p[0] for p in corners)),max(0.,min(p[1] for p in corners)),
         min(float(frame.width),max(p[0] for p in corners)),min(float(frame.height),max(p[1] for p in corners)))
    polygon=tuple(inverse_point(p,frame.width,frame.height,angle) for p in person.mask_polygon_xy) if person.mask_polygon_xy is not None else None
    points=tuple(inverse_point(p,frame.width,frame.height,angle) for p in person.keypoints_xyc) if person.keypoints_xyc is not None else None
    return replace(person,box_xyxy=box,mask_polygon_xy=polygon,keypoints_xyc=points)


class RotationPersonPerception:
    def __init__(self,base,mode='weak',angles=(90,180,270),nms_iou=.6,
                 weak_threshold=.5,probe_interval_s=.5):
        if mode not in ('always','weak','memory') or not 0<nms_iou<=1 or not 0<=weak_threshold<=1 or probe_interval_s<0:
            raise ValueError('Invalid rotation recovery policy')
        if not angles or len(set(angles))!=len(angles) or any(a not in (90,180,270) for a in angles):
            raise ValueError('Distinct cardinal extra rotations required')
        self.base,self.mode,self.angles=base,mode,tuple(angles)
        self.nms_iou,self.weak_threshold,self.interval=nms_iou,weak_threshold,probe_interval_s
        self.last_probe={};self.regions=[];self.region_scope=None
        policy=dict(mode=mode,angles=list(angles),nms_iou=nms_iou,weak_threshold=weak_threshold,
            probe_interval_s=probe_interval_s,extra_classes=['person'],coordinate_mapping='original_image_pixels',
            imgsz=getattr(base,'imgsz',None),confidence=getattr(base,'confidence',None))
        identity=sha256(json.dumps(dict(checkpoint_sha256=base.provenance['sha256'],policy=policy),sort_keys=True).encode()).hexdigest()
        self.provenance=dict(base.provenance,checkpoint_sha256=base.provenance['sha256'],sha256=identity,
            sha256_kind='checkpoint_and_perception_policy',rotation_recovery=policy)
        self.last_detail={}

    def set_recovery_regions(self,frame,regions):
        valid=[]
        for box in regions:
            if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('Invalid recovery attention region')
            valid.append(tuple(box))
        self.regions=valid[:4]
        self.region_scope=(frame.camera_id,frame.stream_epoch,frame.captured_at_s)

    def predict(self,frame,image):
        import cv2
        if image.shape[:2]!=(frame.height,frame.width):raise ValueError('Source image geometry differs from frame reference')
        started=time.perf_counter()
        primary,_=self.base.predict(frame,image)
        native=[p for p in primary.instances if p.label.lower()=='person']
        weak=max((p.score for p in native),default=0.)<self.weak_threshold
        regions=self.regions if self.region_scope==(frame.camera_id,frame.stream_epoch,frame.captured_at_s) else []
        unmatched_memory=any(not any(iou(box,p.box_xyxy)>=.15 for p in native) for box in regions)
        trigger=self.mode=='always' or weak or (self.mode=='memory' and unmatched_memory)
        scope=(frame.camera_id,frame.stream_epoch)
        due=self.mode=='always' or frame.captured_at_s-self.last_probe.get(scope,float('-inf'))>=self.interval-1e-9
        proposals=[(p,0,i) for i,p in enumerate(primary.instances) if p.label.lower()=='person']
        passes=[dict(rotation_deg=0,person_detections=len(native),inference_ms=primary.inference_ms)]
        if trigger and due:
            self.last_probe[scope]=frame.captured_at_s
            for angle in self.angles:
                code={90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
                rotated=cv2.rotate(image,code)
                reference=replace(frame,width=rotated.shape[1],height=rotated.shape[0])
                observed,_=self.base.predict(reference,rotated)
                people=[p for p in observed.instances if p.label.lower()=='person']
                passes.append(dict(rotation_deg=angle,person_detections=len(people),inference_ms=observed.inference_ms))
                for index,person in enumerate(people):
                    restored=inverse_instance(person,frame,angle)
                    if restored.box_xyxy[2]>restored.box_xyxy[0] and restored.box_xyxy[3]>restored.box_xyxy[1]:
                        proposals.append((restored,angle,index))
        kept=[]
        for proposal in sorted(proposals,key=lambda q:(-q[0].score,q[1],q[2])):
            if not any(iou(proposal[0].box_xyxy,p[0].box_xyxy)>self.nms_iou for p in kept):kept.append(proposal)
        # Original non-person evidence is retained; rotation passes add people only.
        other=[p for p in primary.instances if p.label.lower()!='person']
        people=[p[0] for p in kept]
        sources=[0]*len(other)+[p[1] for p in kept]
        result=replace(primary,frame=frame,instances=tuple(other+people),model=self.provenance['sha256'],
                       inference_ms=(time.perf_counter()-started)*1000)
        self.last_detail=dict(passes=passes,source_rotation_deg=sources,triggered=bool(trigger and due),
            weak_primary=weak,unmatched_memory=unmatched_memory,throttled=bool(trigger and not due))
        # Bound scheduler metadata across reconnects/retired camera scopes.
        self.last_probe={key:value for key,value in self.last_probe.items() if frame.captured_at_s-value<=60.}
        return result,self.last_detail
