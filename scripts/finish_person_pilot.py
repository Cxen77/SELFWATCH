"""Wait for the pilot, then compare frozen pretrained/trained weights sequentially."""
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
from selfwatch_v2.overhead_benchmark import benchmark


def compare(baseline,adapted):
    if baseline['split_lock']!=adapted['split_lock'] or baseline['protocol']!=adapted['protocol']:
        raise ValueError('Comparison protocol or frozen split differs')
    keys=lambda report:[(r['image_id'],r['partition'],r['processed_pixel_sha256'],r['ground_truth']) for r in report['frames']]
    if keys(baseline)!=keys(adapted):raise ValueError('Comparison uses different images, pixels or targets')
    return {role:{case:dict(pretrained=baseline['metrics'][role][case],
        adapted=adapted['metrics'][role][case],
        F1_delta=adapted['metrics'][role][case]['F1']-baseline['metrics'][role][case]['F1'])
        for case in baseline['metrics'][role]} for role in ('development','validation')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/wepdtof-pilot'))
    parser.add_argument('--baseline',type=Path,default=Path('weights/v2/yolo26n.pt'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    status=dict(status='waiting_for_training',training=str(args.training),started_utc=datetime.now(timezone.utc).isoformat())
    def persist():
        (args.output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    persist();deadline=time.monotonic()+7200
    try:
        while True:
            experiment=json.loads((args.training/'experiment.json').read_text())
            if experiment['status']=='completed':break
            if experiment['status']=='failed':raise RuntimeError('Training failed; preserve its error log')
            if time.monotonic()>deadline:raise TimeoutError('Training did not finish within the follow-up window')
            time.sleep(10)
        import torch
        # Each complete benchmark finishes before the next model starts. No training overlap.
        reports={}
        for name,weights in (('pretrained',args.baseline),('adapted',args.training/'weights/best.pt')):
            status['status']='benchmark_'+name;persist()
            output=args.output/name
            benchmark(SimpleNamespace(dataset=args.dataset,weights=weights,output=output,
                device='0',budget_ms=50.,max_per_partition=None))
            reports[name]=json.loads((output/'data.json').read_text())
            gc.collect();torch.cuda.empty_cache()
        comparison=compare(reports['pretrained'],reports['adapted'])
        report=dict(schema_version=1,comparison=comparison,
            initial_weights=reports['pretrained']['model'],trained_weights=reports['adapted']['model'],
            protocol=reports['adapted']['protocol'],split_lock=reports['adapted']['split_lock'],
            limitations='Single-seed, small scene-separated pilot. Validation selected the training checkpoint. Final test sealed. No live model promotion or hand/identity accuracy claim.')
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        lines=['# Separate person detector: pretrained versus adapted','',report['limitations'],'',
            '| Split | Case | Pretrained F1 | Adapted F1 | Pretrained recall | Adapted recall |',
            '|---|---|---:|---:|---:|---:|']
        for role,cases in comparison.items():
            for name,row in cases.items():
                lines.append(f'| {role} | {name} | {row["pretrained"]["F1"]:.4f} | {row["adapted"]["F1"]:.4f} | {row["pretrained"]["recall"]:.4f} | {row["adapted"]["recall"]:.4f} |')
        (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status['status']='completed'
    except Exception:
        status['status']='failed';status['error']=traceback.format_exc();raise
    finally:
        status['ended_utc']=datetime.now(timezone.utc).isoformat();persist()


if __name__=='__main__':main()
