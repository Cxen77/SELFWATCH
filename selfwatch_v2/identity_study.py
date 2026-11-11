"""Fixed-input development ablation, not an unlabelled accuracy benchmark.

One YOLO/OSNet pass supplies identical observations/features to both identity
configurations. Cache reuse avoids GPU inference. Pose is added only to selected
review packages, outside identity timings. All real ID scores require labels.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import itertools
import json
from pathlib import Path
import statistics
import time

import numpy as np
from .contracts import FrameRef,Instance,Observation
from .identity import SharedIdentity
from .legacy_guard import LegacyIdentityGuard
from .pipeline import commit_identity
from .replay import load_manifest,replay
from .tracking import LocalTracker,iou


def digest(path):
    value=sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def restore(record):
    instances=[]
    for raw in record['instances']:
        p=dict(raw)
        p['box_xyxy']=tuple(p['box_xyxy'])
        for field in ('mask_polygon_xy','keypoints_xyc'):
            if p.get(field) is not None:p[field]=tuple(map(tuple,p[field]))
        instances.append(Instance(**p))
    return Observation(FrameRef(**record['frame']),tuple(instances),record['model'],record['task'],
                       record['inference_ms'],record.get('keypoint_schema'),record.get('schema_version',1))


class Trial:
    def __init__(self,kind,sources):
        self.kind=kind
        self.trackers={s.camera_id:LocalTracker() if kind=='baseline' else LegacyIdentityGuard(weak_births=kind=='weak_births') for s in sources}
        self.identity=SharedIdentity(allow_overlap=True)
        self.tokens=set();self.gids=set();self.counts=Counter();self.cpu=[]

    def step(self,observation,image,features):
        started=time.perf_counter()
        tracker=self.trackers[observation.frame.camera_id]
        tracks=tracker.update(observation.frame,observation.instances) if self.kind=='baseline' else tracker.preview(observation.frame,observation.instances)
        tracks,metadata=commit_identity(observation,image,tracker,tracks,features,self.identity)
        self.cpu.append((time.perf_counter()-started)*1000)
        visible=[t for t in tracks if not t.get('visual_suppressed') and t.get('global_id') is not None]
        self.counts.update(observations=1,assigned_person_detections=len(visible),
            person_detections=sum(p.label.lower()=='person' for p in observation.instances))
        for t in visible:
            self.tokens.add((t['camera_id'],t['stream_epoch'],t['local_id']))
            self.gids.add(t['global_id'])
        return dict(tracks=tracks,memory=metadata.get('identity_memory',[]),audit=metadata.get('identity_audit',{}))

    def summary(self):
        snap=self.identity.snapshot()
        return dict(**self.counts,distinct_local_tokens=len(self.tokens),distinct_online_global_ids=len(self.gids),
            cross_camera_links=sum(e['association_kind']=='cross_camera' for e in snap['links']),
            protected_returns=len(snap.get('protected_recoveries',[])),
            assigned_detection_fraction=self.counts['assigned_person_detections']/max(1,self.counts['person_detections']),
            cpu_identity_mean_ms=statistics.mean(self.cpu) if self.cpu else 0.,
            cpu_identity_p95_ms=sorted(self.cpu)[int(.95*(len(self.cpu)-1))] if self.cpu else 0.,
            accuracy=None,accuracy_status='No complete manually verified identity ground truth')

    def close(self):
        for tracker in self.trackers.values():
            if hasattr(tracker,'close'):tracker.close()


def run_sample(sample,output,detector,encoder,cache_root=None):
    target=output/sample['sample'];target.mkdir()
    sources=load_manifest(sample['replay_manifest'])
    source_hashes={str(s.path):digest(s.path) for s in sources}
    trials={key:Trial(key,sources) for key in ('baseline','legacy','weak_births')}
    cache=cache_root/sample['sample'] if cache_root else target
    saved_vectors={};records=[];results=[];diagnostics=Counter();max_people=0;crowded=0
    original_cache=None
    if cache_root:
        original_cache=json.loads((cache/'cache.json').read_text(encoding='utf-8'))
        if source_hashes!=original_cache['source_hashes']:
            raise ValueError('Source video differs from cached input')
        for name in ('observations.jsonl','features.npz'):
            if digest(cache/name)!=original_cache['sha256'][name]:raise ValueError('Corrupt perception cache')
        records=[json.loads(line) for line in (cache/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
        vector_file=np.load(cache/'features.npz',allow_pickle=False)
    else:
        vector_file=None
    stream=replay(sources)
    try:
        with (target/'engine.log').open('w',encoding='utf-8') as log, redirect_stdout(log):
            for position,(frame,image) in enumerate(stream):
                if cache_root:
                    if position>=len(records):raise ValueError('Cache lacks source frames')
                    raw=records[position];observation=restore(raw)
                    for field in ('camera_id','stream_epoch','frame_index','captured_at_s','width','height','source'):
                        if getattr(frame,field)!=getattr(observation.frame,field):raise ValueError('Cache frame alignment differs')
                    features={int(i):vector_file[key] for i,key in raw['feature_keys'].items()}
                    reasons=raw['feature_diagnostics']
                else:
                    observation,_=detector.predict(frame,image)
                    features,reasons,_=encoder.extract_all(observation,image)
                    feature_keys={str(i):f'f{position}_i{i}' for i in features}
                    saved_vectors.update({feature_keys[str(i)]:np.asarray(v,dtype=np.float32).copy() for i,v in features.items()})
                    raw=dict(observation.to_dict(),feature_keys=feature_keys,feature_diagnostics=reasons)
                    records.append(raw)
                diagnostics.update(r['reason'] for r in reasons)
                people=[p for p in observation.instances if p.label.lower()=='person']
                max_people=max(max_people,len(people))
                crowded+=any(iou(a.box_xyxy,b.box_xyxy)>=.2 for a,b in itertools.combinations(people,2))
                # All configurations receive this same frame, observation and
                # feature dictionary. No per-tracker detector or ReID calls.
                per_case={key:trial.step(observation,image,features) for key,trial in trials.items()}
                results.append(dict(observation.to_dict(),observation_id=observation.frame.observation_id,
                    tracks=per_case['legacy']['tracks'],tracks_by_case={k:v['tracks'] for k,v in per_case.items()},
                    identity_memory=per_case['legacy']['memory'],identity_audit=per_case['legacy']['audit'],
                    identity_memory_by_case={k:v['memory'] for k,v in per_case.items()},
                    identity_audit_by_case={k:v['audit'] for k,v in per_case.items()}))
    finally:
        stream.close()
        if vector_file is not None:vector_file.close()
        for trial in trials.values():trial.close()
    if cache_root and len(records)!=len(results):raise ValueError('Cache contains additional source frames')
    if not cache_root:
        with (target/'observations.jsonl').open('x',encoding='utf-8') as handle:
            for record in records:handle.write(json.dumps(record,allow_nan=False)+'\n')
        np.savez_compressed(target/'features.npz',**saved_vectors)
        manifest=dict(source_hashes=source_hashes,detector=detector.provenance,reid=encoder.provenance,
            feature_policy='Every eligible crop; scheduled refresh disabled for this controlled identity ablation',
            sha256={name:digest(target/name) for name in ('observations.jsonl','features.npz')})
        (target/'cache.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    else:manifest=original_cache
    # Hash the saved manifest bytes. In-memory model class maps have integer
    # keys, whereas JSON reloads string keys; sorting those differs (2 vs 10).
    cache_digest=digest(cache/'cache.json')
    identities={key:trial.identity.snapshot() for key,trial in trials.items()}
    data=dict(schema_version=1,run_id=f'{output.name}:{sample["sample"]}:{cache_digest[:12]}',
        sample=sample['sample'],name='SELFWATCH fixed-input identity study',frames=results,
        cameras=[s.camera_id for s in sources],tracker_engine='legacy',identity=identities['legacy'],
        identity_by_case=identities,tracking_cases={'legacy':'Original cognitive engine','baseline':'Motion / IoU baseline',
            'weak_births':'Experimental: tentative weak births'},
        pose=dict(providers=[],status='Not used in identity ablation'),source_hashes=source_hashes,
        input_cache=str(cache.resolve()),input_cache_sha256=cache_digest,variants={},
        identity_semantics='Experimental online appearance associations; configurations share identical frozen inputs')
    (target/'data.json').write_text(json.dumps(data,allow_nan=False),encoding='utf-8')
    summary=dict(sample=sample['sample'],observations=len(results),max_detected_people_per_view=max_people,
        overlapping_person_observations=crowded,feature_diagnostics=dict(diagnostics),input_cache_sha256=cache_digest,
        cases={key:trial.summary() for key,trial in trials.items()})
    (target/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def add_review(sample,study,pose):
    import cv2
    from .demo import write_review_assets
    target=study/sample['sample']
    data=json.loads((target/'data.json').read_text(encoding='utf-8'))
    (target/'images').mkdir(exist_ok=False)
    stream=replay(load_manifest(sample['replay_manifest']))
    try:
        for position,(source,image) in enumerate(stream):
            original=data['frames'][position]
            if source.observation_id!=original['observation_id']:raise ValueError('Review frame alignment differs')
            observation=pose.refine(restore(original),image)
            original.update(instances=observation.to_dict()['instances'],keypoint_schema=observation.keypoint_schema,
                            image=f'images/{position:06d}.jpg')
            if not cv2.imwrite(str(target/original['image']),image):raise RuntimeError('Cannot write review image')
    finally:stream.close()
    data['pose']=pose.provenance
    (target/'data.json').write_text(json.dumps(data,allow_nan=False),encoding='utf-8')
    write_review_assets(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=Path('datasets/retailaction-replay-verified/index.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,help='Verify/reuse a previous study; no GPU model loading')
    parser.add_argument('--limit-samples',type=int)
    parser.add_argument('--review-top',type=int,default=1)
    args=parser.parse_args()
    if args.review_top<0 or (args.limit_samples is not None and args.limit_samples<1):raise ValueError('Invalid sample limits')
    args.output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[1]
    code_files=['config.py','selfwatch_v2/identity_study.py','selfwatch_v2/appearance.py',
        'selfwatch_v2/pipeline.py','selfwatch_v2/identity.py','selfwatch_v2/legacy_guard.py',
        'selfwatch_v2/tracking.py','trackers/strongsort_tracker.py','trackers/strack.py',
        'memory/phantom.py','memory/global_identity.py','memory/reasoning.py']
    code_hashes={name:digest(project/name) for name in code_files}
    index=json.loads(args.index.read_text(encoding='utf-8'))
    samples=index['samples'][:args.limit_samples]
    detector=encoder=None
    if args.cache_root is None:
        import torch
        from .yolo import YoloPerception
        from .appearance import AppearanceEncoder
        torch.set_num_threads(2)
        detector=YoloPerception('weights/v2/yolo26n-seg.pt',device='0')
        encoder=AppearanceEncoder('weights/osnet/osnet_x1_0_msmt17.pth')
    started=time.perf_counter();summaries=[]
    for sample in samples:
        summary=run_sample(sample,args.output,detector,encoder,args.cache_root)
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    # Rank developer-review candidates without using annotations or optimizing
    # a test score. Archive-prefix train samples are not a held-out test set.
    ranked=sorted(summaries,key=lambda s:(-s['max_detected_people_per_view'],-s['overlapping_person_observations'],s['sample']))
    worst=sorted(summaries,key=lambda s:(s['cases']['legacy']['assigned_detection_fraction'],s['sample']))
    most_fragmented=sorted(summaries,key=lambda s:(-(s['cases']['baseline']['distinct_local_tokens']-s['cases']['legacy']['distinct_local_tokens']),s['sample']))
    priorities=[ranked[0],worst[0],most_fragmented[0],*ranked] if summaries else []
    selected=list(dict.fromkeys(s['sample'] for s in priorities))[:args.review_top]
    if selected:
        from .wholebody import WholeBody
        pose=WholeBody('weights/v2/rtmw/wholebody133.onnx')
        for sample in samples:
            if sample['sample'] in selected:add_review(sample,args.output,pose)
    report=dict(schema_version=1,dataset=index['dataset'],dataset_revision=index['revision'],split=index['split'],
        selection=index['selection'],method=__doc__,feature_policy='all clean eligible crops, fixed for both cases',
        samples=summaries,review_samples=selected,wall_seconds=time.perf_counter()-started,
        code_sha256=code_hashes,source_changed_during_run=any(digest(project/name)!=value for name,value in code_hashes.items()),
        inference_cache_reused=args.cache_root is not None,accuracy=None,
        limitations=['Unlabelled development diagnostics, not IDF1/HOTA accuracy',
          'A smaller identity count can result from missed tracks; inspect assigned detection fraction too',
          'CPU identity times exclude shared perception, pose and file output; no live FPS claim',
          'Selected irregular retail frames do not represent continuous crowded footage'])
    (args.output/'study.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Fixed-input identity development study','',
        'Identical YOLO observations and OSNet features were supplied to both cases. No labelled identity accuracy is claimed.','',
        '| Sample | Max people/view | Overlap frames | Local tokens B/C/W | Person detections | Assigned detections B/C/W | Cross-view links B/C/W | CPU ms B/C/W |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        b,c,w=(s['cases'][k] for k in ('baseline','legacy','weak_births'))
        values=lambda field:'/'.join(str(v[field]) for v in (b,c,w))
        cpu='/'.join(f'{v["cpu_identity_mean_ms"]:.2f}' for v in (b,c,w))
        lines.append(f'| {s["sample"]} | {s["max_detected_people_per_view"]} | {s["overlapping_person_observations"]} | {values("distinct_local_tokens")} | {b["person_detections"]} | {values("assigned_person_detections")} | {values("cross_camera_links")} | {cpu} |')
    lines.extend(['','B = motion baseline; C = original cognitive engine; W = experimental tentative weak births.',
        '', 'Weak births keep raw detection scores, require multiple observations before a public Person ID, and retain existing territorial/occlusion suppression. This is a development experiment, not the application default.',
        '', 'Review candidates: '+', '.join(selected),
        '', 'Fewer tokens or more links alone do not establish better tracking. Full per-case diagnostics, input hashes, cached features and recorded assignments are in the sample directories.'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
