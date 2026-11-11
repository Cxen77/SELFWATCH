"""Experimental clean LOCAL appearance memory, retaining the original engine.

Association still uses the original tracker's inputs and ownership protections.
The new admission rule prevents scheduled/cached, weak and overlapping crops
from being learned again as fresh local appearance. Not a new tracker or a
claimed implementation of a complete published method. Offline opt-in only.
"""
from collections import Counter
from types import MethodType
from .legacy_guard import LegacyIdentityGuard
from .tracking import iou


class ContinuityGuard(LegacyIdentityGuard):
    def __init__(self,policy='clean_memory',**kwargs):
        if policy not in ('clean_memory','strict_margin','clean_margin'):raise ValueError('Unknown continuity policy')
        super().__init__(**kwargs);self.policy=policy
        self.continuity_audit={};self._admissions={}

    def _reset(self,frame):
        super()._reset(frame)
        if self.policy in ('strict_margin','clean_margin'):self.engine.tracker.ambiguity_margin=.03

    def _wrap(self,track):
        if getattr(track,'_continuity_admission_wrapped',False):return
        original=track.update;guard=self
        def update(owner,box,score,embedding=None,**kwargs):
            key=tuple(map(float,box));allowed=guard._admissions.get(key,False)
            guard._counter['local_gallery_admitted' if allowed else 'local_gallery_held']+=1
            return original(box,score,embedding if allowed else None,**kwargs)
        track.update=MethodType(update,track)
        track._continuity_admission_wrapped=True

    def commit(self,frame,instances,image,embeddings):
        self._counter=Counter();self._admissions={}
        if self.policy!='strict_margin':
            boxes=Counter(tuple(map(float,p.box_xyxy)) for _,p in self._people)
            for index,person in self._people:
                key=tuple(map(float,person.box_xyxy))
                clean=(index in embeddings and person.score>=.7 and boxes[key]==1 and
                    not any(other_index!=index and iou(person.box_xyxy,other.box_xyxy)>.3
                            for other_index,other in self._people))
                self._admissions[key]=clean
            for track in self.engine.tracker.tracks:self._wrap(track)
        result=super().commit(frame,instances,image,embeddings)
        if self.policy!='strict_margin':
            for track in self.engine.tracker.tracks:
                # Newborn tracks can inherit a preview's cached feature before
                # they ever receive a trusted observation. Keep their position,
                # but do not retain that borrowed/zero-valued appearance.
                if track.total_hits==1 and track.time_since_update==0 and not self._admissions.get(tuple(map(float,track.box)),False):
                    track.embedding=None;track._emb_history.clear();track.memory_update_allowed=False
                    self._counter['untrusted_birth_gallery_cleared']+=1
                    token=self.engine.global_id_manager.get_global_id(track.local_id)
                    for row in result:
                        if row['local_id']==token:row['memory_update_allowed']=False
        self.continuity_audit=dict(self._counter)
        self.audit['continuity_policy']=dict(name=self.policy,confidence_floor=.7,overlap_limit=.3,
            fresh_feature_required=True,margin=.03 if self.policy!='clean_memory' else 0.,counts=self.continuity_audit)
        return result
