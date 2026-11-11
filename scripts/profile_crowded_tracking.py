"""Offline crowded public-frame profile, including tracking and persistence costs.

No capture/display FPS claim. Source frames are never omitted from replay.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modes',nargs='+',default=['detailed','fast','fast','detailed'])
    parser.add_argument('--pose-device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--frames',type=int,default=12)
    parser.add_argument('--no-reid-graph',action='store_true')
    args=parser.parse_args()
    from selfwatch_v2.gpu_runtime import activate
    if args.pose_device=='cuda':activate()
    import cv2
    import torch
    from selfwatch_v2.studio import Studio
    from selfwatch_v2.contracts import FrameRef
    from selfwatch_v2.pipeline import process
    from selfwatch_v2.legacy_guard import LegacyIdentityGuard
    from selfwatch_v2.identity import SharedIdentity
    from selfwatch_v2.store import ObservationStore
    torch.set_num_threads(2)
    root=Path.cwd();studio=Studio(root,pose_device=args.pose_device,reid_graph=not args.no_reid_graph)
    with redirect_stdout(io.StringIO()):models=studio._load_models(True)
    dataset=root/'datasets/wildtrack-pilot-v1'
    manifest=json.loads((dataset/'manifest.json').read_text())
    count=min(args.frames,len(manifest['frames']))
    images=[]
    for number,index in enumerate(manifest['frames'][:count]):
        path=dataset/f'Image_subsets/C1/{index:08d}.png';image=cv2.imread(str(path))
        image=cv2.resize(image,(1280,720))
        frame=FrameRef('C1','crowded-profile',number,number*.5,0,1280,720,str(path.resolve()),'WILDTRACK annotated 2Hz')
        images.append((frame,image))
    output_dir=args.output.with_suffix('');output_dir.mkdir(parents=True,exist_ok=False)
    class NoPose:
        provenance={'schema':None}
        def refine(self,o,i):return o
    results=[]
    for trial,mode in enumerate(args.modes):
        detector=models['detector'];detector.include_masks=mode=='detailed'
        pose=models['pose'] if mode=='detailed' else NoPose()
        if mode=='balanced':
            from selfwatch_v2.pose_schedule import ScheduledPose
            pose=ScheduledPose(models['pose'],interval_s=.5,max_people=2)
        if mode not in ('detailed','fast','balanced'):raise ValueError('Unknown profile mode')
        encoder=models['appearance'];encoder.states.clear();encoder.attention.clear()
        tracker=LegacyIdentityGuard(log_dir=output_dir/f'trial-{trial}-memory')
        timings=[];records=[];people=[];save_times=[];started=time.perf_counter()
        try:
            with redirect_stdout(io.StringIO()),ObservationStore(output_dir/f'trial-{trial}.sqlite') as store:
                for obs,image,tracks,diagnostics,timing in process(iter(images),detector,pose,{'C1':tracker},encoder,
                        SharedIdentity(allow_overlap=True),parallel=args.pose_device=='cpu'):
                    tick=time.perf_counter();store.append(obs)
                    ok,_=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                    if not ok:raise RuntimeError('JPEG encoding failed')
                    json.dumps(dict(observation=obs.to_dict(),tracks=tracks,timings=timing),allow_nan=False)
                    save_times.append((time.perf_counter()-tick)*1000)
                    timings.append(timing);records.append(tracks)
                    people.append(sum(p.label.lower()=='person' for p in obs.instances))
        finally:tracker.close()
        wall=time.perf_counter()-started
        means={key:statistics.mean(t.get(key,0) for t in timings) for key in
            ('detector_ms','pose_ms','reid_ms','cognitive_engine_ms','association_ms','processing_age_ms')}
        result=dict(mode=mode,frames=len(timings),wall_seconds=wall,observations_per_s=len(timings)/wall,
            mean_ms=means,p95_processing_age_ms=sorted(t['processing_age_ms'] for t in timings)[max(0,int(len(timings)*.95)-1)],
            mean_persistence_ms=statistics.mean(save_times),people=dict(min=min(people),max=max(people),mean=statistics.mean(people)),
            observed_tracks=sum(len(t) for t in records),pose_device=args.pose_device,
            max_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
            reid_graph=not args.no_reid_graph,
            identity_signature_sha256=hashlib.sha256(json.dumps([[(t['instance_index'],t['local_id'],t['global_id']) for t in row] for row in records]).encode()).hexdigest())
        results.append(result);print(json.dumps(result),flush=True)
        args.output.write_text(json.dumps(dict(trials=results,protocol='Warmed same public C1 frames at 1280x720; original 2Hz timestamps; one camera; all detections and original cognitive memory; SQLite + JPEG + JSON included; images predecoded; no frontend or capture timing; not accuracy validation',
            input_manifest=str(dataset/'manifest.json'),pose=models['pose'].provenance,reid=models['appearance'].provenance),indent=2),encoding='utf-8')
    studio.close()

if __name__=='__main__':main()
