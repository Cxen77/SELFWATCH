"""Alternate serial/overlapped compute in one warmed process; report all trials.

Includes decoding, detection, ReID, tracking, association and pose; excludes
SQLite/JPEG export and model construction. This is offline, not live latency.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time
import torch

from selfwatch_v2.appearance import AppearanceEncoder
from selfwatch_v2.identity import SharedIdentity
from selfwatch_v2.pipeline import process
from selfwatch_v2.replay import load_manifest,replay
from selfwatch_v2.tracking import LocalTracker
from selfwatch_v2.wholebody import WholeBody
from selfwatch_v2.yolo import YoloPerception


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('runs/overlap-benchmark.json'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    sources=load_manifest(args.manifest)
    detector=YoloPerception('weights/v2/yolo26n-seg.pt',device='0')
    pose=WholeBody('weights/v2/rtmw/wholebody133.onnx')
    appearance=AppearanceEncoder('weights/osnet/osnet_x1_0_msmt17.pth')

    def run(parallel,limit=None):
        appearance.states,appearance.attention={},{}
        identity=SharedIdentity(allow_overlap=True)
        stream=replay(sources)
        processed=process(itertools.islice(stream,limit) if limit else stream,detector,pose,
            {s.camera_id:LocalTracker() for s in sources},appearance,identity,parallel=parallel)
        timings,signature=[],[]
        started=time.perf_counter()
        try:
            for observation,_,tracks,_,timing in processed:
                timings.append(timing)
                signature.append((observation.frame.observation_id,
                    [(t['local_id'],t['global_id']) for t in tracks]))
        finally:
            processed.close();stream.close()
        elapsed=time.perf_counter()-started
        return dict(parallel=parallel,frames=len(timings),seconds=elapsed,
            aggregate_frames_per_second=len(timings)/elapsed,
            processing_age_p50_ms=statistics.median(t['processing_age_ms'] for t in timings),
            pose_p50_ms=statistics.median(t['pose_ms'] for t in timings)),signature

    run(True,10)
    trials=[];reference=None
    for parallel in (False,True,False,True):
        trial,signature=run(parallel)
        if reference is None:reference=signature
        if signature!=reference:
            raise AssertionError('Scheduling changed frame order or global assignments')
        trials.append(trial)
        print(json.dumps(trial),flush=True)
    report=dict(scope=__doc__,trials=trials,identical_assignments=True,
        serial_median_seconds=statistics.median(t['seconds'] for t in trials if not t['parallel']),
        overlap_median_seconds=statistics.median(t['seconds'] for t in trials if t['parallel']),
        peak_torch_reserved_mb=torch.cuda.max_memory_reserved()/1024**2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
