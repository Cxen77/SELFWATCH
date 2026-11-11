"""Bridge the ORIGINAL cognitive engine to v2 observations, not a second tracker.

The original engine's 'global IDs' are CAMERA-LOCAL protected person tokens here.
Only SharedIdentity allocates outward-facing, cross-camera Person IDs. StrongSORT,
phantoms, warm retrieval, fingerprints, ownership and the shadow audit all run.
"""
from collections import defaultdict, deque
from hashlib import sha256
from types import SimpleNamespace
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import config

from engine.pipeline import SelfWatchPipeline, STATE_ACTIVE, STATE_THINKING, STATE_PHANTOM
from trackers.strongsort_tracker import StrongSORTTracker
from memory.contradiction import ContradictionDetector
from memory.gait import GaitSignature
from .tracking import iou


class NoImplicitInference:
    def extract_batch(self, *args):
        raise RuntimeError("Legacy engine attempted unscheduled ReID; embeddings must be injected")

    def detect(self, *args, **kwargs):
        raise RuntimeError("Legacy engine attempted a second detector; YOLO results must be injected")


class LegacyIdentityGuard:
    name = "original_strongsort_cognitive_engine"

    def __init__(self, log_dir=None, reference_hz=30., weak_births=False):
        if not math.isfinite(reference_hz) or reference_hz <= 0:
            raise ValueError("Positive finite reference frequency required")
        self.log_dir, self.reference_hz = log_dir, reference_hz
        self.weak_births=weak_births
        self.camera = self.epoch = None
        self.engine = None
        self.now = self.first_time = self.previous_time = None
        self.tick = 0
        self.last_seen, self.history = {}, defaultdict(lambda: deque(maxlen=64))
        self.memory, self._sources = [], {}
        self.intervals=deque(maxlen=32)

    def _reset(self, frame):
        if self.engine:
            self.engine.event_logger.close()
        self.now = self.first_time = frame.captured_at_s
        self.previous_time, self.tick = None, 0
        self.last_seen.clear()
        self.history.clear()
        self.intervals.clear()
        self.gait=GaitSignature(fps_estimate=self.reference_hz)
        self.contradictions=ContradictionDetector(check_interval=1)
        log_dir = str(self.log_dir / sha256(frame.camera_id.encode()).hexdigest()[:12]) if self.log_dir else "logs"
        guard = NoImplicitInference()
        tracker=StrongSORTTracker(appearance_weight=config.TRACKER_APPEARANCE_WEIGHT,
            high_thresh=config.TRACKER_HIGH_THRESH,low_thresh=config.TRACKER_LOW_THRESH,
            iou_thresh=config.TRACKER_IOU_THRESH,max_cosine_dist=config.TRACKER_MAX_COSINE_DIST,
            max_lost=config.TRACKER_MAX_LOST,confirm_threshold=config.TRACKER_CONFIRM_THRESHOLD,
            embedding_history=config.TRACKER_EMBEDDING_HISTORY,
            min_quality_score=config.TRACKER_MIN_QUALITY_SCORE,allow_weak_births=self.weak_births)
        self.engine = SelfWatchPipeline(guard, guard, tracker,
            enable_debug_overlay=False, preserve_frame_size=True,
            event_clock=lambda: self.now, event_ticks=True,
            log_dir=log_dir, event_logging=self.log_dir is not None,
            enable_retail=False, enable_shelf_change=False, enable_tracer=False,
            min_person_size=(16, 16))
        # Replay can be sparse: decay by capture-time intervals, not 30 calls.
        self.engine.warm_memory.DECAY_INTERVAL = 1
        self.engine.reasoning.reference_hz = self.reference_hz
        self.camera, self.epoch = frame.camera_id, frame.stream_epoch

    def recovery_regions(self, frame):
        """Recent missing-owner regions for detector attention, never evidence.

        Read the previous committed snapshot without advancing the engine. An
        epoch reconnect, other camera or stale memory cannot request a probe.
        """
        if (frame.camera_id,frame.stream_epoch)!=(self.camera,self.epoch):
            return []
        regions=[]
        for record in self.memory:
            seen=record.get('last_seen_s')
            box=record.get('predicted_box_xyxy',[])
            if (record['observed'] or record['state'] not in ('HELD','THINKING','PHANTOM')
                or seen is None or not 0<=frame.captured_at_s-seen<=2.
                or record.get('prediction_confidence',1.)<.3 or len(box)!=4):
                continue
            if not all(math.isfinite(x) for x in box):continue
            x1,y1,x2,y2=box
            clipped=(max(0.,x1),max(0.,y1),min(float(frame.width),x2),min(float(frame.height),y2))
            if clipped[2]>clipped[0] and clipped[3]>clipped[1]:
                regions.append((seen,clipped))
        return [box for _,box in sorted(regions,key=lambda row:-row[0])[:4]]

    def preview(self, frame, instances):
        """ROI scheduling hints only. The original engine makes ALL associations."""
        if self.camera is not None and frame.camera_id != self.camera:
            raise ValueError("A camera-local engine cannot consume another camera")
        if self.engine is None or frame.stream_epoch != self.epoch:
            self._reset(frame)
        if self.previous_time is not None and frame.captured_at_s <= self.previous_time:
            raise ValueError("Camera timestamps must strictly increase")
        self.now = frame.captured_at_s
        if self.previous_time is not None:
            self.intervals.append(self.now-self.previous_time)
        target = max(self.tick + 1, int(round((self.now-self.first_time)*self.reference_hz)) + 1)
        self.delta = target-self.tick
        self.tick = target
        self._people = [(i,p) for i,p in enumerate(instances)
                        if p.label.lower() == "person" and
                        p.box_xyxy[2]-p.box_xyxy[0] >= 16 and p.box_xyxy[3]-p.box_xyxy[1] >= 16]
        old = self.engine.tracker.tracks
        self._sources = {}
        if old and self._people:
            costs = np.ones((len(old),len(self._people)),dtype=np.float32)
            for r,t in enumerate(old):
                predicted = t.smooth_box + np.tile(t.vel * self.delta, 2)
                for c,(_,p) in enumerate(self._people):
                    costs[r,c] = 1-iou(predicted,p.box_xyxy)
            rows,cols = linear_sum_assignment(costs)
            self._sources = {self._people[c][0]:old[r] for r,c in zip(rows,cols) if costs[r,c] < .9}
        result=[]
        for i,_ in self._people:
            t=self._sources.get(i)
            token=self.engine.global_id_manager.get_global_id(t.local_id) if t else None
            result.append(dict(instance_index=i,local_id=token if token is not None else -(frame.frame_index*65536+i+1),
                               hits=t.total_hits if t else 1))
        return result

    def commit(self, frame, instances, image, embeddings):
        vectors=[]
        for i,_ in self._people:
            previous=self._sources.get(i)
            cached=previous.embedding if previous is not None else None
            vector=embeddings.get(i,cached)
            vectors.append(vector if vector is not None else np.zeros(512,dtype=np.float32))
        detections=SimpleNamespace(boxes=[list(p.box_xyxy) for _,p in self._people],
            scores=[p.score for _,p in self._people],class_ids=[0]*len(self._people))
        _,stats=self.engine.process_frame_with_precomputed(image,detections,
            np.asarray(vectors,dtype=np.float32).reshape(-1,512),
            frame_delta=self.delta,frame_index=frame.frame_index)
        self.previous_time = self.now
        observed=[t for t in self.engine.tracker.tracks if t.time_since_update == 0]
        assignment={}
        if observed and self._people:
            costs=np.array([[1-iou(t.box,p.box_xyxy) for _,p in self._people] for t in observed])
            rows,cols=linear_sum_assignment(costs)
            assignment={self._people[c][0]:observed[r] for r,c in zip(rows,cols) if costs[r,c] < .9}
        output=[]
        self._observed_tokens=set()
        frozen=set(stats['frozen_gids'])
        suppressed={int(gid):reason for gid,reason in stats['suppressed_entries']}
        for i,t in assignment.items():
            token=self.engine.global_id_manager.get_global_id(t.local_id)
            if token is None:
                continue
            if getattr(t,'weak_birth',False) and not t.is_confirmed:
                # Keep a raw weak detection tentative internally. Do not issue
                # an outward-facing Person ID until independent observations.
                self._observed_tokens.add(token)
                self.last_seen[token]=self.now
                continue
            # An identity can have several latent tracker fragments, never two
            # observed boxes in this output. Keep the stronger observation.
            existing=next((v for v in output if v['local_id']==token),None)
            if existing is not None:
                if instances[existing['instance_index']].score >= instances[i].score:
                    continue
                output.remove(existing)
            self._observed_tokens.add(token)
            self.last_seen[token]=self.now
            box=instances[i].box_xyxy
            self.history[token].append([self.now,(box[0]+box[2])/2,(box[1]+box[3])/2])
            state="FROZEN" if token in frozen else "ACTIVE" if t.is_confirmed else "TENTATIVE"
            # ByteTrack's spatial-only stage and body-shape quality rules can
            # hold LOCAL memory even for a clean new detection. Keep the new
            # cross-view policy, with original hard overlap/anchor vetoes.
            anchor=getattr(t,'_anchor_embedding',None)
            vector=embeddings.get(i)
            anchor_veto=(anchor is not None and vector is not None and t.total_hits>100
                         and float(np.dot(anchor,vector))<.55)
            cross_view_allowed=(token not in self.engine.occlusion_manager.frozen_gids
                and getattr(t,'_identity_freeze_frames',0)<=0 and not anchor_veto)
            output.append(dict(instance_index=i,camera_id=frame.camera_id,stream_epoch=frame.stream_epoch,
                local_id=int(token),tracker_local_id=int(t.local_id),hits=int(t.total_hits),
                trajectory=list(self.history[token]),status="observed",protection_state=state,
                memory_update_allowed=bool(t.memory_update_allowed and token not in frozen),
                cross_view_update_allowed=bool(cross_view_allowed),
                visual_suppressed=token in suppressed,suppression_reason=suppressed.get(token),
                association_method=t.last_assoc_method,coordinate_space="image_pixels"))
        records={}
        for t in self.engine.tracker.tracks:
            token=self.engine.global_id_manager.get_global_id(t.local_id)
            if token is not None:
                records[token]=dict(state="TENTATIVE" if t.is_tentative and t.time_since_update==0 else "HELD",
                                    predicted_box_xyxy=t.predicted_box.tolist())
        for token,record in self.engine._id_states.items():
            if record['state']==STATE_THINKING:
                records[token]=dict(state="THINKING",predicted_box_xyxy=list(record.get('last_box',[])))
            elif record['state']==STATE_PHANTOM:
                records[token]=dict(state="PHANTOM")
        for token,phantom in self.engine.phantom_tracker.phantoms.items():
            records[token]=dict(state="PHANTOM",predicted_box_xyxy=phantom.position.tolist(),
                                prediction_confidence=float(phantom.confidence))
        for token in self.engine.warm_memory.identities:
            if token not in self._observed_tokens:
                records[token]=dict(state="WARM")
        for track in output:
            records[track['local_id']]=dict(state=track['protection_state'])
        self.memory=[dict(local_id=int(token),observed=token in self._observed_tokens,
            last_seen_s=self.last_seen.get(token),
            age_s=max(0.,self.now-self.last_seen[token]) if token in self.last_seen else None,
            **record) for token,record in sorted(records.items())]
        for token in set(self.history)-set(records):
            del self.history[token]
        for token in set(self.last_seen)-set(records):
            self.last_seen.pop(token,None)
        for token in set(self.engine.topology._track_zones)-set(records):
            self.engine.topology.clear_track(token)
        self.audit=dict(captured_at_s=self.now,virtual_tick=self.tick,reference_hz=self.reference_hz,
            phantom_count=self.engine.phantom_tracker.count,warm_count=self.engine.warm_memory.count,
            frozen_local_ids=sorted(map(int,frozen)),suppressed_entries=stats['suppressed_entries'],
            ownership=self.engine.ownership_sm.get_summary(),
            metrics=self.engine.metrics.get_summary(),
            recent_rebinds=self.engine.global_id_manager.get_rebind_log()[-10:])
        self.audit['contradictions']=self.contradictions.tick(observed)
        live_lids={t.local_id for t in self.engine.tracker.tracks}
        for lid in list(self.contradictions._embedding_history):
            if lid not in live_lids:
                self.contradictions.clear_track(lid)
        # FFT gait is invalid on sparse selected frames. Never invent a stride
        # from the pilot's irregular 3 Hz samples or bridge a detection gap.
        median=float(np.median(self.intervals)) if self.intervals else 0.
        dense=(len(self.intervals)>=3 and 0<median<=.1 and
               max(abs(dt-median) for dt in self.intervals)<=median*.1)
        visible_tokens={t['local_id'] for t in output}
        for token in list(self.gait._windows):
            if not dense or token not in visible_tokens:
                self.gait.clear_track(token)
        if dense:
            self.gait.fps_estimate=1/median
            for track in output:
                if track['protection_state']=='ACTIVE':
                    self.gait.update(track['local_id'],instances[track['instance_index']].box_xyxy)
        self.audit['gait']=dict(sampling_valid=dense,
            signatures_available=sum(self.gait.get_signature(token) is not None for token in visible_tokens),
            used_for_identity=False,reason="advisory only; uniform >=10 Hz samples required")
        self.audit['scene_topology']=dict(ready=self.engine.topology.is_ready,
            used_for_cross_camera_identity=False,coordinate_space="per_camera_image_grid")
        self.audit['experimental_weak_births']=self.weak_births
        return sorted(output,key=lambda t:t['instance_index'])

    def close(self):
        if self.engine:
            self.engine.event_logger.close()
