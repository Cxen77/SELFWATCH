"""Compare real person embeddings and warm timing before enabling CUDA replay."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
import torch
from selfwatch_v2.appearance import AppearanceEncoder
from selfwatch_v2.cuda_reid import GraphedReID


def main():
    torch.set_num_threads(2)
    with redirect_stdout(io.StringIO()):encoder=AppearanceEncoder('weights/osnet/osnet_x1_0_msmt17.pth')
    extractor=encoder.extractor
    data=json.loads(Path('runs/wildtrack-tracking-v1/predictions.json').read_text())
    rows=data['frames']
    crops=[]
    for row in rows:
        camera=row['frame']['camera_id'];index=row['frame']['frame_index']
        image=cv2.imread(f'datasets/wildtrack-pilot-v1/Image_subsets/{camera}/{index:08d}.png')
        for p in row['instances']:
            if p['label']!='person' or p['score']<.5:continue
            x1,y1,x2,y2=map(int,p['box_xyxy']);h,w=image.shape[:2]
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            if min(x2-x1,y2-y1)<16:continue
            crops.append(cv2.resize(image[y1:y2,x1:x2],(128,256)))
            if len(crops)>=32:break
        if len(crops)>=32:break
    outputs={};report=[]
    original=extractor._model
    for mode in ('eager','graph'):
        if mode=='graph':
            extractor._model=GraphedReID(original,extractor._device,extractor._half,extractor._reid_stream)
        for n in (1,2,3,4,5,8,12,16):
            capacity=next(s for s in (1,2,4,8,16) if s>=n)
            batch=crops[:n]+[crops[n-1]]*(capacity-n)
            for _ in range(2):extractor.extract_batch(batch)
            values=[]
            for _ in range(5):
                start=time.perf_counter();vectors=extractor.extract_batch(batch)[:n];values.append((time.perf_counter()-start)*1000)
            if mode=='eager':outputs[n]=vectors.copy()
            else:
                reference=outputs[n];delta=float(np.max(np.abs(reference-vectors)))
                cosine=float(np.min(np.sum(reference*vectors,axis=1)))
                if delta>.001 or cosine<.9999:raise RuntimeError(f'Embedding mismatch for batch {n}')
            report.append(dict(mode=mode,people=n,padded_batch=capacity,median_ms=statistics.median(values),
                max_abs_error=None if mode=='eager' else delta,min_cosine=None if mode=='eager' else cosine))
    # Change all crop rows after capture: a graph must compute new evidence,
    # never replay cached person features from an earlier observation.
    graphed=extractor._model
    updates=[]
    for n in (1,2,4,8,16):
        batch=list(reversed(crops[-n:]))
        extractor._model=original;reference=extractor.extract_batch(batch)
        extractor._model=graphed;actual=extractor.extract_batch(batch)
        delta=float(np.max(np.abs(reference-actual)));cosine=float(np.min(np.sum(reference*actual,axis=1)))
        if delta>.001 or cosine<.9999:raise RuntimeError('Graph reused stale person features')
        updates.append(dict(batch=n,max_abs_error=delta,min_cosine=cosine))
    # Compare bucket padding against unpadded eager inference too.
    extractor._model=original
    padding=[]
    for n in (3,5,12):
        reference=extractor.extract_batch(crops[:n]);delta=float(np.max(np.abs(reference-outputs[n])))
        cosine=float(np.min(np.sum(reference*outputs[n],axis=1)))
        if delta>.001 or cosine<.9999:raise RuntimeError('Padding changed embeddings materially')
        padding.append(dict(people=n,max_abs_error=delta,min_cosine=cosine))
    path=Path('runs/reid-graph-validation-20261108.json')
    path.write_text(json.dumps(dict(trials=report,padding=padding,changed_inputs=updates,passed=True,
        protocol='Real public WILDTRACK person crops; same OSNet weights and preprocessing; CUDA FP16; 5 timed calls after 2 warmups per shape. Embedding equivalence is not tracking accuracy validation.',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,model=encoder.provenance,
        max_allocated_mb=torch.cuda.max_memory_allocated()/1024**2),indent=2),encoding='utf-8')
    print(path);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
