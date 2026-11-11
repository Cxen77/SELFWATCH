"""Continuous, native-frame MOT17 identity experiments with frozen perception."""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import statistics
import time

import numpy as np


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def dataset(root,sequence):
    manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    setting=manifest['selection']['sequences'][sequence]
    rows=[r for r in manifest['files'] if r['path'].startswith(sequence+'/')]
    if len(rows)!=setting['frames']+2:raise ValueError('Incomplete sequence manifest')
    for row in rows:
        path=root/row['path']
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('Dataset checksum mismatch')
    return setting,digest(manifest_path)


def frame_image(root,sequence,index):
    import cv2
    from .contracts import FrameRef
    path=root/sequence/'img1'/f'{index:06d}.jpg';image=cv2.imread(str(path))
    if image is None or image.shape[:2]!=(1080,1920):raise ValueError('Unexpected native frame geometry')
    image=cv2.resize(image,(1280,720))
    frame=FrameRef(sequence,'mot17-continuity-v1',index-1,(index-1)/30,0,1280,720,str(path.resolve()),'official native MOT17 frame index / 30Hz')
    return frame,image


def cache(args):
    from .studio import Studio
    setting,manifest_hash=dataset(args.dataset,args.sequence)
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(2)
    studio=Studio(Path.cwd(),reid_graph=True)
    with redirect_stdout(io.StringIO()):models=studio._load_models(False)
    detector=models['detector'];detector.include_masks=False;encoder=models['appearance']
    vectors={};counts=[];forward=[];started=time.perf_counter()
    source_files=['selfwatch_v2/yolo.py','selfwatch_v2/appearance.py','selfwatch_v2/cuda_reid.py','selfwatch_v2/mot_continuity.py']
    with (args.output/'perception.jsonl').open('w',encoding='utf-8') as out:
        for index in range(1,setting['frames']+1):
            frame,image=frame_image(args.dataset,args.sequence,index)
            obs,_=detector.predict(frame,image)
            obs=replace(obs,instances=tuple(p for p in obs.instances if p.label.lower()=='person'))
            features,reasons,elapsed=encoder.extract_all(obs,image)
            for number,vector in features.items():vectors[f'{index}:{number}']=vector
            out.write(json.dumps(dict(observation=obs.to_dict(),appearance_diagnostics=reasons),allow_nan=False)+'\n')
            counts.append(len(obs.instances));forward.append(elapsed)
            if index%100==0:print(args.sequence,index,'/',setting['frames'],'people',len(obs.instances),flush=True)
    np.savez_compressed(args.output/'features.npz',**vectors)
    config=dict(sequence=args.sequence,role=setting['role'],frames=setting['frames'],fps=30,
        dataset_manifest_sha256=manifest_hash,detector=detector.provenance,reid=encoder.provenance,
        input_wh=[1280,720],detector_imgsz=640,detector_confidence=.25,feature_policy='all_clean_eligible',
        perception_sha256=digest(args.output/'perception.jsonl'),features_sha256=digest(args.output/'features.npz'),
        source_sha256={name:digest(name) for name in source_files},people_mean=statistics.mean(counts),people_max=max(counts),
        cache_wall_seconds=time.perf_counter()-started,reid_forward_mean_ms=statistics.mean(forward),
        protocol='Native JPEGs resized to 1280x720; every source frame; same detector and all clean eligible embeddings cached for all policies; no GT inference input. Not live adaptive scheduling throughput.')
    (args.output/'cache.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    studio.close();print('Perception cache frozen',args.output,flush=True)


def replay(args):
    import cv2
    from .identity_study import restore
    from .identity import SharedIdentity
    from .legacy_guard import LegacyIdentityGuard
    from .pipeline import commit_identity
    from .mot_metrics import evaluate_mot
    setting,manifest_hash=dataset(args.dataset,args.sequence)
    config=json.loads((args.cache/'cache.json').read_text())
    if (config['sequence']!=args.sequence or config['dataset_manifest_sha256']!=manifest_hash or
        config['perception_sha256']!=digest(args.cache/'perception.jsonl') or config['features_sha256']!=digest(args.cache/'features.npz')):
        raise ValueError('Cached perception source/hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    features=np.load(args.cache/'features.npz',allow_pickle=False)
    cases=args.cases
    if setting['role']=='evaluation':
        if args.policy is None:raise ValueError('Evaluation requires a frozen development policy')
        frozen=json.loads(args.policy.read_text())
        if frozen['selected_case'] not in cases:raise ValueError('Evaluation must include selected development policy')
        expected=frozen['tracker_sources']
        if any(digest(path)!=hash_ for path,hash_ in expected.items()):raise ValueError('Tracker changed after policy freeze')
        for key,value in frozen['perception_signature'].items():
            actual=config['detector']['sha256'] if key=='detector_sha256' else config['reid']['sha256'] if key=='reid_sha256' else config[key]
            if actual!=value:raise ValueError('Evaluation perception differs from development')
    results={}
    for case in cases:
        tracker=LegacyIdentityGuard()
        if case!='baseline':
            from .continuity_guard import ContinuityGuard
            tracker=ContinuityGuard(policy=case)
        identity=SharedIdentity(allow_overlap=True)
        path=args.output/'trackers'/case/'data'/f'{args.sequence}.txt';path.parent.mkdir(parents=True)
        timings=[];audit_counts={};emitted=0
        with path.open('w',encoding='utf-8') as output,(args.output/f'{case}-engine.log').open('w',encoding='utf-8') as log:
            try:
                for index,line in enumerate((args.cache/'perception.jsonl').open(encoding='utf-8'),1):
                    row=json.loads(line);observation=restore(row['observation'])
                    if observation.frame.frame_index!=index-1:raise ValueError('Non-continuous frame cache')
                    _,image=frame_image(args.dataset,args.sequence,index)
                    vectors={i:features[f'{index}:{i}'] for i in range(len(observation.instances)) if f'{index}:{i}' in features}
                    tick=time.perf_counter()
                    with redirect_stdout(log):
                        tracks=tracker.preview(observation.frame,observation.instances)
                        tracks,metadata=commit_identity(observation,image,tracker,tracks,vectors,identity)
                    timings.append((time.perf_counter()-tick)*1000)
                    for key,value in getattr(tracker,'continuity_audit',{}).items():audit_counts[key]=audit_counts.get(key,0)+value
                    for track in tracks:
                        if track.get('status','observed')!='observed' or track.get('visual_suppressed'):continue
                        person=observation.instances[track['instance_index']]
                        x1,y1,x2,y2=[v*1.5 for v in person.box_xyxy]
                        gid=track['global_id']
                        output.write(f'{index},{gid},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{person.score:.6f},-1,-1,-1\n')
                        emitted+=1
            finally:tracker.close()
        result=evaluate_mot(args.dataset,args.output/'trackers',args.sequence,setting['frames'],case)
        result.update(cpu_identity_mean_ms=statistics.mean(timings),cpu_identity_p95_ms=float(np.percentile(timings,95)),
            emitted_observations=emitted,continuity_audit=audit_counts,predictions_sha256=digest(path))
        results[case]=result;print(case,json.dumps(result['summary']),flush=True)
        report=dict(sequence=args.sequence,role=setting['role'],frames=setting['frames'],results=results,
            cache_sha256=digest(args.cache/'cache.json'),dataset_manifest_sha256=manifest_hash,
            frozen_policy_sha256=digest(args.policy) if args.policy else None,
            scope='Complete single-camera MOT17 training sequences, our detector and original cognitive tracker. Standard pinned TrackEval preprocessing/metrics. No cross-camera identity or retail-transfer claim.')
        (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    features.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['cache','replay'])
    parser.add_argument('--dataset',type=Path,default=Path('datasets/mot17-continuity-native-v1'))
    parser.add_argument('--sequence',choices=['MOT17-02-FRCNN','MOT17-09-FRCNN'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache',type=Path)
    parser.add_argument('--cases',nargs='+',choices=['baseline','clean_memory','strict_margin','clean_margin'],default=['baseline']);parser.add_argument('--policy',type=Path)
    args=parser.parse_args();(cache if args.mode=='cache' else replay)(args)

if __name__=='__main__':main()
