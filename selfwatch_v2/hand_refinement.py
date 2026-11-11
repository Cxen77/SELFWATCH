"""Explicit-device research hand refiner. No live pipeline uses it by default."""
from hashlib import sha256
from pathlib import Path
import numpy as np


def hand_roi(points,width,height):
    points=np.asarray(points,dtype=float)
    if points.shape!=(21,3) or not np.isfinite(points).all():return None
    usable=points[points[:,2]>=.45]
    if len(usable)<8:return None
    lo=usable[:,:2].min(axis=0);hi=usable[:,:2].max(axis=0)
    center=(lo+hi)/2;size=np.maximum(hi-lo,16.)
    lo=np.maximum(center-size/2,0);hi=np.minimum(center+size/2,[width,height])
    if np.any(hi-lo<=1):return None
    return dict(box=tuple(map(float,(*lo,*hi))),score=float(np.median(usable[:,2])))


class HandPose:
    def __init__(self,path,device='cpu'):
        from rtmlib import RTMPose
        import onnxruntime as ort
        path=Path(path).resolve()
        self.model=RTMPose(str(path),model_input_size=(256,256),to_openpose=False,backend='onnxruntime',device='cpu')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.log_severity_level=3
        if device=='cpu':providers=['CPUExecutionProvider']
        elif device=='cuda':providers=[('CUDAExecutionProvider',dict(device_id=0,gpu_mem_limit=1024*1024**2,
            arena_extend_strategy='kNextPowerOfTwo',cudnn_conv_algo_search='HEURISTIC',
            cudnn_conv_use_max_workspace='0',use_tf32='0')),'CPUExecutionProvider']
        else:raise ValueError('Explicit cpu or cuda hand device required')
        self.model.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        if device=='cuda' and 'CUDAExecutionProvider' not in self.model.session.get_providers():
            raise RuntimeError('CUDA hand inference unavailable; no silent CPU fallback')
        self.provenance=dict(checkpoint=str(path),sha256=sha256(path.read_bytes()).hexdigest(),
            providers=self.model.session.get_providers(),runtime=ort.__version__,threads=2,input_wh=[256,256],
            actual_model_batch=1,requested_device=device,gpu_arena_limit_mb=1024 if device=='cuda' else None,
            status='Research-only, not enabled in live tracking')

    def predict(self,image,box):
        points,scores=self.model(image,bboxes=[box])
        if points.shape!=(1,21,2) or scores.shape!=(1,21):raise ValueError('Unexpected hand model topology')
        result=np.column_stack((points[0],scores[0]))
        if not np.isfinite(result).all():raise ValueError('Nonfinite hand output')
        return result


CpuHandPose=HandPose # Retain the original CPU-default research API.
