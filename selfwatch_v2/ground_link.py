"""Offline, causal linking of synchronized, calibrated person tracklets."""
from itertools import combinations
import math

import numpy as np


class GroundCamera:
    """Map an undistorted image to a shared floor in metres."""
    def __init__(self, image_from_floor, width=1920, height=1080):
        matrix=np.asarray(image_from_floor,dtype=float)
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or np.linalg.cond(matrix)>1e10:
            raise ValueError('Invalid floor homography')
        self.floor_from_image=np.linalg.inv(matrix)
        self.width,self.height=width,height

    def foot(self, box):
        x1,y1,x2,y2=box
        # Edge-clipped boxes may not contain the foot or its horizontal centre.
        if not (1<x1<x2<self.width-1 and 0<=y1<y2<self.height-1):
            return None
        xy=self.floor_from_image@np.array([(x1+x2)/2,y2,1.])
        if xy[2]<=1e-9:
            return None
        return xy[:2]/xy[2]


def mutual_pairs(left,right,radius,margin):
    if not left or not right:
        return []
    a,b=list(left),list(right)
    matrix=np.linalg.norm(np.array(list(left.values()))[:,None]-np.array(list(right.values()))[None,:],axis=2)
    rows,cols=np.argmin(matrix,axis=0),np.argmin(matrix,axis=1)
    pairs=[]
    for i,j in enumerate(cols):
        dist=matrix[i,j]
        if rows[j]!=i or dist>radius:
            continue
        alt_a=np.partition(matrix[i],1)[1] if len(b)>1 else math.inf
        alt_b=np.partition(matrix[:,j],1)[1] if len(a)>1 else math.inf
        if min(alt_a,alt_b)-dist>=margin:
            pairs.append((a[i],b[j],float(dist)))
    return pairs


class GroundLinker:
    """Confirm two adjacent synchronized sightings before joining camera-local IDs."""
    def __init__(self,radius,margin,max_gap=.75,min_cosine=None):
        if not all(math.isfinite(x) and x>0 for x in (radius,margin,max_gap)):
            raise ValueError('Positive finite geometry settings required')
        self.radius,self.margin,self.max_gap=radius,margin,max_gap
        if min_cosine is not None and not -1<=min_cosine<=1:
            raise ValueError('Invalid appearance corroboration floor')
        self.min_cosine=min_cosine
        self.parents={};self.groups={};self.ordinal={};self.prior=set()
        self.events=[];self.last=-math.inf

    def root(self,key):
        while self.parents[key]!=key:
            key=self.parents[key]
        return key

    def update(self,timestamp,views,features=None):
        if not math.isfinite(timestamp) or timestamp<=self.last:
            raise ValueError('Expected strictly increasing synchronized bundles')
        keys={key for view in views.values() for key in view}
        for camera,view in views.items():
            for key in view:
                if key[0]!=camera:
                    raise ValueError('Camera-scoped key does not match view')
                if key not in self.parents:
                    self.parents[key]=key
                    self.groups[key]={key}
                    self.ordinal[key]=len(self.ordinal)+1
        points={key:point for view in views.values() for key,point in view.items() if point is not None}
        eligible=[{key:p for key,p in view.items() if p is not None} for view in views.values()]
        candidates=sorted((distance,a,b) for x,y in combinations(eligible,2)
            for a,b,distance in mutual_pairs(x,y,self.radius,self.margin))
        if self.min_cosine is not None:
            features=features or {}
            candidates=[(distance,a,b) for distance,a,b in candidates if a in features and b in features
                        and float(np.dot(features[a],features[b]))>=self.min_cosine]
        confirmed=self.prior if timestamp-self.last<=self.max_gap else set()
        self.prior={frozenset((a,b)) for _,a,b in candidates}
        self.last=timestamp
        for distance,a,b in candidates:
            if frozenset((a,b)) not in confirmed:
                continue
            x,y=sorted((self.root(a),self.root(b)),key=self.ordinal.get)
            if x==y:
                continue
            left,right=self.groups[x],self.groups[y]
            if {k[0] for k in left}&{k[0] for k in right}:
                continue
            if any(np.linalg.norm(points[p]-points[q])>self.radius
                   for p in left if p in points for q in right if q in points):
                continue
            self.parents[y]=x
            left.update(self.groups.pop(y))
            self.events.append(dict(captured_at_s=timestamp,pair=[a,b],distance_m=distance,
                retained_id=self.ordinal[x],joined_id=self.ordinal[y]))
        return {key:self.ordinal[self.root(key)] for key in keys}
