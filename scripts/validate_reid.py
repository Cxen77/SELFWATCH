"""Check strict legacy OSNet loading and CPU/FP16 GPU embedding agreement."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from reid import EmbeddingExtractor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('weights/osnet/osnet_x1_0_msmt17.pth'))
    parser.add_argument('--output',type=Path,default=Path('runs/reid-validation.json'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    image=cv2.imread(str(args.image))
    if image is None:
        raise ValueError('Validation image missing')
    crops=[cv2.resize(image,(128,256)),cv2.resize(cv2.flip(image,1),(128,256))]
    cpu=EmbeddingExtractor(args.weights,device='cpu',strict_weights=True,input_size=(256,128),max_batch=16)
    gpu=EmbeddingExtractor(args.weights,device='cuda:0',strict_weights=True,input_size=(256,128),max_batch=16)
    a,b=cpu.extract_batch(crops),gpu.extract_batch(crops)
    similarities=np.sum(a*b,axis=1)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(similarities)<.995:
        raise AssertionError('CPU/GPU features diverged')
    if not np.allclose(np.linalg.norm(b,axis=1),1,atol=1e-5):
        raise AssertionError('Embeddings are not normalized')
    report=dict(cpu_load=cpu.load_report,gpu_load=gpu.load_report,shape=list(b.shape),
        per_crop_cpu_gpu_cosine=similarities.tolist(),input_height_width=[256,128],
        scope='Numerical feature agreement on image fixtures; not ReID accuracy')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
