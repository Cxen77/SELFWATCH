"""Output-equivalence check on matching recorded observations, not pose accuracy."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np


def compare(cpu,gpu):
    left=json.loads((cpu/'data.json').read_text());right=json.loads((gpu/'data.json').read_text())
    if len(left['frames'])!=len(right['frames']) or left['pose']['sha256']!=right['pose']['sha256']:
        raise ValueError('Different model or observation count')
    visible=[];all_distances=[];scores=[];people=0;id_differences=0
    for a,b in zip(left['frames'],right['frames']):
        fields=('camera_id','stream_epoch','frame_index','captured_at_s','source','width','height')
        if any(a['frame'][key]!=b['frame'][key] for key in fields):raise ValueError('Not the same source observations')
        if len(a['instances'])!=len(b['instances']):raise ValueError('Detector outputs differ')
        for p,q in zip(a['instances'],b['instances']):
            if p['label']!=q['label'] or not np.allclose(p['box_xyxy'],q['box_xyxy'],atol=1e-4,rtol=0):
                raise ValueError('Person crops or detector classes differ')
            if p['keypoints_xyc'] is None:
                if q['keypoints_xyc'] is not None:raise ValueError('Different pose budget')
                continue
            if q['keypoints_xyc'] is None:raise ValueError('Missing GPU pose')
            x,y=np.asarray(p['keypoints_xyc']),np.asarray(q['keypoints_xyc'])
            if x.shape!=(133,3) or y.shape!=x.shape:raise ValueError('Invalid pose schema')
            distances=np.linalg.norm(x[:,:2]-y[:,:2],axis=1);all_distances.extend(distances)
            visible.extend(distances[(x[:,2]>=.45)&(y[:,2]>=.45)]);scores.extend(abs(x[:,2]-y[:,2]));people+=1
        id_differences+=int([(t['instance_index'],t['global_id']) for t in a['tracks']]!=[(t['instance_index'],t['global_id']) for t in b['tracks']])
    if not visible:raise ValueError('No comparable confident landmarks')
    metrics=dict(person_observations=people,visible_joints=len(visible),all_joints=len(all_distances),
        visible_p95_pixel_delta=float(np.percentile(visible,95)),visible_max_pixel_delta=float(np.max(visible)),
        all_max_pixel_delta=float(np.max(all_distances)),max_score_delta=float(np.max(scores)),
        identity_frame_differences=id_differences)
    passed=people>=50 and len(visible)>=100 and metrics['visible_max_pixel_delta']<=1 and metrics['max_score_delta']<=.01
    timings={}
    for name,root in (('cpu',cpu),('gpu',gpu)):
        summary=json.loads((root/'summary.json').read_text());rows=json.loads((root/'timings.json').read_text())
        timings[name]=dict(frames=summary['frames'],wall_seconds=summary['wall_seconds'],
            offline_observations_per_s=summary['frames']/summary['wall_seconds'],
            pose_mean_ms=float(np.mean([r['pose_ms'] for r in rows])),providers=summary['providers'])
    return dict(passed=passed,cpu_run=str(cpu),gpu_run=str(gpu),metrics=metrics,timings=timings,
        warning='Numerical equivalence and offline workload timing only; not independent pose accuracy or live multi-camera throughput')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record-validated-runtime',action='store_true');args=parser.parse_args()
    result=compare(args.cpu,args.gpu);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.record_validated_runtime:
        if not result['passed']:raise ValueError('Runtime equivalence check did not pass')
        root=Path(__file__).resolve().parents[1];metadata=json.loads((root/'.runtime/pose-gpu-runtime.json').read_text())
        (root/'.runtime/pose-gpu-validated.json').write_text(json.dumps(dict(validated=True,
            wheel_sha256=metadata['sha256'],report=str(args.output.resolve()),
            report_sha256=sha256(args.output.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
