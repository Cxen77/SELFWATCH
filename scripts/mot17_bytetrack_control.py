"""Untuned standard ByteTrack reference on exactly the frozen person detections."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from selfwatch_v2.mot_metrics import evaluate_mot
from selfwatch_v2.mot_continuity import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads((args.cache/'cache.json').read_text())
    if digest(args.cache/'perception.jsonl')!=config['perception_sha256']:raise ValueError('Perception cache changed')
    args.output.mkdir(parents=True,exist_ok=False)
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.engine.results import Boxes
    import ultralytics
    parameters=dict(track_high_thresh=.5,track_low_thresh=.1,new_track_thresh=.5,track_buffer=300,match_thresh=.8,fuse_score=True)
    tracker=BYTETracker(SimpleNamespace(**parameters));case='bytetrack';seq=config['sequence']
    path=args.output/'trackers'/case/'data'/f'{seq}.txt';path.parent.mkdir(parents=True)
    times=[]
    with path.open('w',encoding='utf-8') as out:
        for index,line in enumerate((args.cache/'perception.jsonl').open(encoding='utf-8'),1):
            obs=json.loads(line)['observation']
            rows=np.array([[*p['box_xyxy'],p['score'],0] for p in obs['instances']],dtype=np.float32).reshape(-1,6)
            tick=time.perf_counter();tracks=tracker.update(Boxes(rows,orig_shape=(720,1280)));times.append((time.perf_counter()-tick)*1000)
            for track in tracks:
                x1,y1,x2,y2,gid,score,_,_=track
                # Use the associated measured detector box, matching the legacy
                # export; filtered Kalman boxes would change the perception input.
                instance=int(track[-1]);person=obs['instances'][instance]
                x1,y1,x2,y2=[v*1.5 for v in person['box_xyxy']]
                out.write(f'{index},{int(gid)},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{score:.6f},-1,-1,-1\n')
    result=evaluate_mot(Path('datasets/mot17-continuity-native-v1'),args.output/'trackers',seq,config['frames'],case)
    result.update(cpu_tracking_mean_ms=float(np.mean(times)),ultralytics=ultralytics.__version__,parameters=parameters)
    report=dict(sequence=seq,role=config['role'],frames=config['frames'],results={case:result},
        cache_sha256=digest(args.cache/'cache.json'),scope='Untuned motion-only reference; same detections, no appearance model or legacy memory. Not live promotion.')
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()
