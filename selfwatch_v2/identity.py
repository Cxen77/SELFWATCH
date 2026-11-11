"""Capture-time association around the old registry and cognitive galleries.

Appearance links are hypotheses, not calibrated identity probabilities. The
controller allows active overlapping views, abstains on close alternatives and
records provisional-link revisions. It never globally merges two established
groups solely because their mean embeddings are similar.
"""
import numpy as np
from multicam.global_registry import GlobalMultiCameraIdentityManager
from memory.cognitive import CognitiveMemory


class SharedIdentity:
    def __init__(self, similarity=.78, margin=.06, confirmations=2,
                 active_ttl_s=2., memory_ttl_s=300., allow_overlap=False, max_identities=200):
        if not -1<=similarity<=1 or not 0<=margin<=2 or confirmations<1 or min(active_ttl_s,memory_ttl_s)<=0 or max_identities<1:
            raise ValueError("Invalid identity association policy")
        self.now = float("-inf")
        self.registry = GlobalMultiCameraIdentityManager(clock=lambda:self.now)
        self.threshold, self.margin, self.confirmations = similarity, margin, confirmations
        self.active_ttl, self.memory_ttl = active_ttl_s, memory_ttl_s
        self.allow_overlap, self.max_identities = allow_overlap, max_identities
        self.seen, self.identities, self.pending, self.aliases = {}, {}, {}, {}
        self.views, self.events = {}, []
        self.established = set()
        self.reciprocal = {}
        self.epochs = {}
        self.held, self.protections, self.lifecycle_events = {}, {}, []

    def _canonical(self, gid):
        while gid in self.aliases:
            gid=self.aliases[gid]
        return gid

    def synchronize_memory(self, frame, records):
        """Reserve protected local owners without pretending phantoms are visible."""
        if frame.captured_at_s < self.now:
            raise ValueError("Global identity updates must follow capture time")
        self.now=frame.captured_at_s
        states={(frame.camera_id,(frame.stream_epoch,r['local_id'])):r for r in records}
        self.protections[frame.camera_id]=states
        for key in list(self.held):
            gid=self._canonical(self.held[key])
            if (key[0]==frame.camera_id and key not in states) or self.now-self.identities.get(gid,-1e9)>self.memory_ttl:
                del self.held[key]
        for key in list(self.seen):
            if key[0]!=frame.camera_id or key not in states or states[key]['observed']:
                continue
            gid=self.registry.get_global_id(*key)
            if gid is not None:
                self.held[key]=gid
                self.registry.unregister_local_track(*key)
                self.registry.move_to_dormant(gid,key[0],frame.frame_index)
            del self.seen[key]
            self.pending.pop(key,None)

    def memory_snapshot(self, frame, records):
        result=[]
        for record in records:
            key=(frame.camera_id,(frame.stream_epoch,record['local_id']))
            gid=self.registry.get_global_id(*key)
            if gid is None:
                gid=self.held.get(key)
            result.append(dict(record,global_id=self._canonical(gid) if gid is not None else None))
        return result

    def _expire(self, frame):
        camera, epoch = frame.camera_id, frame.stream_epoch
        old_epoch = self.epochs.get(camera)
        for key,last in list(self.seen.items()):
            if self.now-last>self.active_ttl or (old_epoch is not None and old_epoch!=epoch and key[0]==camera):
                gid = self.registry.get_global_id(key[0],key[1])
                if key in self.protections.get(key[0],{}) and key[1][0]==self.epochs.get(key[0]):
                    if gid is not None:
                        self.held[key]=gid
                self.registry.unregister_local_track(key[0],key[1])
                if gid is not None:
                    self.registry.move_to_dormant(gid,key[0],frame.frame_index)
                del self.seen[key]
                self.pending.pop(key,None)
        self.epochs[camera] = epoch

    def _candidates(self, vector, camera, own):
        candidates = []
        for gid,last in self.identities.items():
            if gid==own or gid in self.aliases or self.now-last>self.memory_ttl:
                continue
            members = self.registry.get_local_tracks(gid)
            # A different local fragment cannot steal a protected lost owner.
            if any(key[0]==camera and self._canonical(value)==gid for key,value in self.held.items()):
                continue
            if any(cam==camera for cam,_ in members):
                continue
            if members and not self.allow_overlap:
                continue
            gallery=[]
            for memory in self.views.values():
                record=memory.get_identity_embeddings(gid)
                if record:
                    gallery.extend([record["stable"],record["recent"],record["best_quality"],*record["gallery"]])
            if gallery:
                score=max(float(np.dot(vector,prototype)) for prototype in gallery)
                candidates.append(dict(global_id=gid, cosine=score))
        return sorted(candidates,key=lambda c:(-c["cosine"],c["global_id"]))[:3]

    def update(self, frame, tracks, embeddings, instances):
        if frame.captured_at_s < self.now:
            raise ValueError("Global identity updates must follow capture time")
        self.now=frame.captured_at_s
        self._expire(frame)
        self.reciprocal={pair:{key:value for key,value in evidence.items() if self.now-value<3.}
                         for pair,evidence in self.reciprocal.items()
                         if any(self.now-value<3. for value in evidence.values())}
        if frame.camera_id not in self.views:
            self.views[frame.camera_id]=CognitiveMemory()
        memory=self.views[frame.camera_id]
        output=[]
        for track in tracks:
            token=(frame.stream_epoch,track["local_id"])
            key=(frame.camera_id,token)
            own=self.registry.get_global_id(*key)
            held=self.held.pop(key,None)
            recovered=False
            if own is None and held is not None:
                held=self._canonical(held)
                if held in self.identities and self.now-self.identities[held]<=self.memory_ttl:
                    own=self.registry.register_local_track(frame.camera_id,token,global_id=held)
                    recovered=True
                    self.lifecycle_events.append(dict(captured_at_s=self.now,camera_id=frame.camera_id,
                        stream_epoch=frame.stream_epoch,local_id=track['local_id'],global_id=own,
                        event="protected_owner_return",method="original_cognitive_engine"))
            if own is None:
                if len(self.identities)>=self.max_identities:
                    removable=[gid for gid in self.identities if not self.registry.is_active_anywhere(gid)]
                    if not removable:
                        output.append(dict(track,global_id=None,identity_status="capacity_abstention",candidates=[]))
                        continue
                    oldest=min(removable,key=self.identities.get)
                    self.registry.forget_inactive(oldest)
                    del self.identities[oldest]
                    for view in self.views.values():
                        view.forget_identity_embeddings(oldest)
                own=self.registry.register_local_track(frame.camera_id,token)
                self.identities[own]=self.now
            self.seen[key]=self.now
            vector=embeddings.get(track["instance_index"])
            status="existing_link" if own in self.established or len(self.registry.get_local_tracks(own))>1 else "local_visit"
            if recovered:
                status="protected_recovery"
            candidates=[]
            admit=vector is not None
            if vector is not None:
                vector=np.asarray(vector,dtype=np.float32)
                norm=np.linalg.norm(vector)
                if vector.ndim!=1 or not np.isfinite(vector).all() or norm<1e-8:
                    raise ValueError("Identity embeddings must be finite nonzero vectors")
                vector=vector/norm
                candidates=self._candidates(vector,frame.camera_id,own)
                established=own in self.established or len(self.registry.get_local_tracks(own))>1
                if candidates and not established:
                    best=candidates[0]
                    separation=best["cosine"]-(candidates[1]["cosine"] if len(candidates)>1 else -1.)
                    if best["cosine"]>=self.threshold and separation>=self.margin:
                        proposal=self.pending.get(key)
                        count=proposal[1]+1 if proposal and proposal[0]==best["global_id"] and self.now-proposal[2]<3. else 1
                        self.pending[key]=(best["global_id"],count,self.now)
                        # Two clean reciprocal view observations can corroborate
                        # provisional identities without reducing the threshold.
                        pair=tuple(sorted((own,best["global_id"])))
                        support=self.reciprocal.setdefault(pair,{})
                        support[key]=self.now
                        reciprocal=(self.allow_overlap and len({k[0] for k in support})>=2
                                    and max(support.values())-min(support.values())>=.15)
                        evidence_count=max(count,len(support) if reciprocal else 0)
                        status="candidate_link"
                        if evidence_count>=self.confirmations:
                            target=best["global_id"]
                            source_camera=self.registry.get_memory_entry(target).last_seen_camera
                            kind="cross_camera" if source_camera!=frame.camera_id else "same_camera_recovery"
                            previous=own
                            self.registry.unregister_local_track(*key)
                            self.registry.move_to_dormant(previous,frame.camera_id,frame.frame_index)
                            self.registry.register_local_track(frame.camera_id,token,global_id=target)
                            self.aliases[previous]=target
                            if kind=="cross_camera":
                                self.registry.record_cross_camera_match()
                            own=target
                            self.established.add(target)
                            status="appearance_link"
                            event=dict(captured_at_s=self.now,camera_id=frame.camera_id,stream_epoch=frame.stream_epoch,
                                local_id=track["local_id"],old_global_id=previous,global_id=target,
                                cosine=best["cosine"],evidence_count=evidence_count,method="appearance_only_unvalidated",
                                association_kind=kind,source_camera=source_camera,
                                evidence_type="reciprocal_views" if reciprocal else "repeated_track_observations")
                            self.events.append(event)
                            self.pending.pop(key,None)
                    else:
                        self.pending.pop(key,None)
                        if best["cosine"]>=self.threshold:
                            status="ambiguous"
                if established:
                    record=self.registry.get_memory_entry(own)
                    if record.stable_embedding is not None and float(np.dot(record.stable_embedding,vector))<.4:
                        status="appearance_conflict"
                        admit=False
                if admit:
                    memory.update_active_identity(own,vector,instances[track["instance_index"]].score)
            self.registry.update_observation(frame.camera_id,own,embedding=vector if admit else None,
                box=list(instances[track["instance_index"]].box_xyxy),frame_index=frame.frame_index)
            self.identities[own]=self.now
            output.append(dict(track,global_id=own,identity_status=status,candidates=candidates,
                               appearance_fresh=vector is not None))
        return output

    def snapshot(self):
        return dict(registry=self.registry.get_stats(), aliases=self.aliases,
            protected_recoveries=self.lifecycle_events,
            links=self.events, policy=dict(similarity=self.threshold,margin=self.margin,
                confirmations=self.confirmations,allow_overlap=self.allow_overlap,
                reciprocal_window_s=3.,reciprocal_min_span_s=.15,
                active_ttl_s=self.active_ttl,memory_ttl_s=self.memory_ttl,max_identities=self.max_identities),
            warning="Appearance-only experimental associations; no calibrated cross-view geometry or identity accuracy claim")
