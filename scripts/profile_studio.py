"""Warm, alternating-order source-workspace profile on the same paired clip."""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--output',type=Path,default=Path('runs/studio-fps-comparison.json'));args=parser.parse_args()
    base=f'http://127.0.0.1:{args.port}'
    def get(path):
        with urllib.request.urlopen(base+path,timeout=10) as response:return json.load(response)
    token=get('/api/bootstrap')['token']
    def run(mode,limit=64):
        if get('/api/state')['status'] in ('loading','running','stopping'):raise RuntimeError('Another session is active')
        payload=json.dumps(dict(kind='sample',max_frames=limit,performance=mode,pose=True)).encode()
        request=urllib.request.Request(base+'/api/start',data=payload,
            headers={'Content-Type':'application/json','X-Selfwatch-Token':token})
        with urllib.request.urlopen(request,timeout=10) as response:started=json.load(response)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            state=get('/api/state')
            if state['run_id']!=started['run_id']:raise RuntimeError('Session replaced during profiling')
            if state['status']=='failed':raise RuntimeError(state['error'])
            if state['status'] in ('completed','stopped'):break
            time.sleep(.25)
        else:raise RuntimeError('Profile timed out; stop the session in the workspace')
        root=Path('runs')/started['run_id'];summary=json.loads((root/'summary.json').read_text())
        config=json.loads((root/'run.json').read_text())
        result=dict(mode=mode,run_id=started['run_id'],frames=summary['frames'],wall_seconds=summary['wall_seconds'],
            offline_observations_per_s=summary['frames']/summary['wall_seconds'],recent_fps=state['fps'],
            stage_mean_ms=state['stage_mean_ms'],pose=config['pose'],identity_links=summary['cross_camera_links'])
        print(json.dumps(result),flush=True);return result
    warmup=run('detailed',8)
    trials=[run(mode) for mode in ('detailed','fast','fast','detailed')]
    modes={mode:dict(mean_offline_observations_per_s=statistics.mean(t['offline_observations_per_s'] for t in trials if t['mode']==mode),
        trial_values=[t['offline_observations_per_s'] for t in trials if t['mode']==mode]) for mode in ('detailed','fast')}
    report=dict(warmup=warmup,trials=trials,summary=modes,
        protocol='Same 64 timestamped paired observations, one visible person per view, warmed models, detailed/fast/fast/detailed order',
        exclusions='Offline source-workspace throughput, not camera capture or three-camera/crowded-store capacity. Accuracy is not measured.')
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(modes,indent=2),flush=True)


if __name__=='__main__':main()
