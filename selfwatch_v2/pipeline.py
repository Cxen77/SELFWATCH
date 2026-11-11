"""Bounded CPU/GPU overlap; identity is committed in capture order.

Only one thread calls GPU models. The CPU pose worker owns its pose session.
No observations are dropped in replay. At most two pose jobs are in flight.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time


def timed_pose(pose, observation, image, submitted):
    started=time.perf_counter()
    result=pose.refine(observation,image)
    return result, dict(pose_queue_ms=(started-submitted)*1000,
                        pose_ms=(time.perf_counter()-started)*1000,
                        pose_diagnostics=dict(getattr(pose,'last_diagnostics',{})))


def commit_identity(observation,image,tracker,tracks,embeddings,identity):
    """Shared identity stage for live scheduling and fixed-input experiments."""
    guarded=hasattr(tracker,'commit')
    timings={}
    if guarded:
        started=time.perf_counter()
        tracks=tracker.commit(observation.frame,observation.instances,image,embeddings)
        timings['cognitive_engine_ms']=(time.perf_counter()-started)*1000
        embeddings={t['instance_index']:embeddings[t['instance_index']] for t in tracks
                    if t['cross_view_update_allowed'] and not t['visual_suppressed']
                    and t['instance_index'] in embeddings}
    started=time.perf_counter()
    if guarded:
        identity.synchronize_memory(observation.frame,tracker.memory)
    tracks=identity.update(observation.frame,tracks,embeddings,observation.instances)
    if guarded:
        timings['identity_memory']=identity.memory_snapshot(observation.frame,tracker.memory)
        timings['identity_audit']=tracker.audit
    timings['association_ms']=(time.perf_counter()-started)*1000
    return tracks,timings


def process(stream, detector, pose, trackers, appearance, identity, parallel=True):
    pending=deque()
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="selfwatch-cpu-pose") if parallel else None
    try:
        for frame,image in stream:
            started=time.perf_counter()
            tracker=trackers[frame.camera_id]
            if hasattr(detector,'set_recovery_regions'):
                regions=tracker.recovery_regions(frame) if hasattr(tracker,'recovery_regions') else []
                detector.set_recovery_regions(frame,regions)
            observation,detail=detector.predict(frame,image)
            timings=dict(detector_ms=observation.inference_ms)
            if isinstance(detail,dict) and 'passes' in detail:
                timings['perception_diagnostics']=detail
            tick=time.perf_counter()
            guarded=hasattr(tracker,"commit")
            tracks=(tracker.preview(frame,observation.instances) if guarded
                    else tracker.update(frame,observation.instances))
            timings["local_tracking_ms"]=(time.perf_counter()-tick)*1000
            tick=time.perf_counter()
            embeddings,diagnostics,inference_ms=appearance.extract(observation,image,tracks)
            timings.update(reid_ms=(time.perf_counter()-tick)*1000,reid_forward_ms=inference_ms)
            tracks,identity_times=commit_identity(observation,image,tracker,tracks,embeddings,identity)
            timings.update(identity_times)
            submitted=time.perf_counter()
            future=executor.submit(timed_pose,pose,observation,image,submitted) if executor else None
            payload=(future,None if future else timed_pose(pose,observation,image,submitted),
                     image,tracks,diagnostics,timings,started)
            pending.append(payload)
            if len(pending)>=(2 if parallel else 1):
                yield resolve(pending.popleft())
        while pending:
            yield resolve(pending.popleft())
    finally:
        if executor:
            executor.shutdown(wait=True,cancel_futures=True)


def resolve(payload):
    future,result,image,tracks,diagnostics,timings,started=payload
    observation,pose_times=future.result() if future else result
    timings.update(pose_times)
    timings["processing_age_ms"]=(time.perf_counter()-started)*1000
    return observation,image,tracks,diagnostics,timings
