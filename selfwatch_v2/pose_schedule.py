"""Bound optional landmarks without skipping detection or identity observations."""
from collections import OrderedDict
from dataclasses import replace
import math


class ScheduledPose:
    def __init__(self,base,interval_s=.5,max_people=2,max_scopes=16):
        if not math.isfinite(interval_s) or interval_s<=0 or type(max_people) is not int or max_people<1 or max_scopes<1:
            raise ValueError('Positive pose scheduling limits required')
        self.base,self.interval,self.max_people,self.max_scopes=base,interval_s,max_people,max_scopes
        self.states=OrderedDict();self.last_diagnostics={}
        self.provenance=dict(base.provenance,scheduling=dict(interval_s=interval_s,max_people_per_call=max_people,
            selection='rotating current detection indices per camera/epoch; no identity claim',
            stale_keypoints='never reused',max_scopes=max_scopes))

    def refine(self,observation,image):
        frame=observation.frame;key=(frame.camera_id,frame.stream_epoch);now=frame.captured_at_s
        state=self.states.setdefault(key,dict(last=-math.inf,seen=-math.inf,cursor=0))
        if now<state['seen']:raise ValueError('Pose scheduler received backward camera time')
        state['seen']=now;self.states.move_to_end(key)
        while len(self.states)>self.max_scopes:self.states.popitem(last=False)
        people=[i for i,p in enumerate(observation.instances) if p.label.lower()=='person']
        self.last_diagnostics=dict(eligible_people=len(people),posed_people=0,reason='interval',interval_s=self.interval)
        if not people or now-state['last']<self.interval-1e-6:
            if not people:self.last_diagnostics['reason']='no_people'
            return observation
        start=state['cursor']%len(people)
        chosen=(people[start:]+people[:start])[:self.max_people]
        subset=replace(observation,instances=tuple(observation.instances[i] for i in chosen))
        refined=self.base.refine(subset,image)
        if len(refined.instances)!=len(chosen):raise ValueError('Pose refiner changed instance count')
        instances=list(observation.instances)
        for index,person in zip(chosen,refined.instances):instances[index]=person
        state.update(last=now,cursor=start+len(chosen))
        self.last_diagnostics.update(posed_people=len(chosen),reason='fresh_subset',instance_indices=chosen)
        return replace(refined,instances=tuple(instances))
