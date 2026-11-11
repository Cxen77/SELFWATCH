"""Run one frozen detection-only pilot. Never replace the portal's current models."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import traceback
import cv2
from selfwatch_v2.overhead_data import contained


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('weights/v2/yolo26n.pt'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args()
    manifest_path=args.dataset/'export_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    audit=json.loads(args.audit.read_text())
    if audit['export_manifest_sha256']!=sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Audit belongs to another training export')
    if manifest['task']!='detect':raise ValueError('Person-box pilot requires detection targets')
    # Verify file integrity and forbidden partitions before GPU initialization.
    for item in manifest['items']:
        if item['partition'] not in ('development','validation'):raise ValueError('Forbidden training partition')
        labels=contained(args.dataset,item['labels']);image=contained(args.dataset,item['image'])
        if sha256(labels.read_bytes()).hexdigest()!=item['label_sha256']:
            raise ValueError('Training labels changed since export')
        pixels=cv2.imread(str(image))
        if pixels is None or sha256(pixels.tobytes()).hexdigest()!=item['pixel_sha256']:
            raise ValueError('Training pixels changed since export')
    recipe=json.loads((args.dataset/'recipe.json').read_text())
    recipe.update(project=str(args.output.parent.resolve()),name=args.output.name,exist_ok=False)
    if args.output.exists():raise ValueError('Training output already exists; preserve prior experiments')
    args.output.mkdir(parents=True)
    lock=dict(status='starting',started_utc=datetime.now(timezone.utc).isoformat(),
        export_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        audit_sha256=sha256(args.audit.read_bytes()).hexdigest(),
        initial_weights=dict(path=str(args.weights.resolve()),sha256=sha256(args.weights.read_bytes()).hexdigest()),
        recipe=recipe,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol='One bounded pilot, single seed; best checkpoint selected by validation fitness. Final test sealed.',
        limitation='Person boxes only. No hand, mask, item, ReID or depth supervision; no live model promotion.')
    def persist():
        (args.output/'experiment.json').write_text(json.dumps(lock,indent=2,default=str),encoding='utf-8')
    persist()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        torch.set_num_threads(2)
        lock['hardware']=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
            ultralytics=ultralytics.__version__,torch_threads=torch.get_num_threads())
        model=YOLO(str(args.weights.resolve()))
        if model.task!='detect':raise ValueError('Separate detector checkpoint required')
        # The pre-created directory contains the lock; allow only this exact directory.
        recipe['exist_ok']=True;lock['status']='training';persist()
        model.train(**recipe)
        lock['status']='completed'
        lock['checkpoints']={name:dict(sha256=sha256((args.output/'weights'/name).read_bytes()).hexdigest(),
            bytes=(args.output/'weights'/name).stat().st_size) for name in ('best.pt','last.pt')}
    except Exception:
        lock['status']='failed';lock['error']=traceback.format_exc();raise
    finally:
        lock['ended_utc']=datetime.now(timezone.utc).isoformat();persist()


if __name__=='__main__':main()
