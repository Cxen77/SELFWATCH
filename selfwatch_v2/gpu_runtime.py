"""Select the isolated CUDA-12 ORT before any CPU ORT module is imported."""
import json
from pathlib import Path
import sys


def activate(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1]
    metadata=json.loads((root/'.runtime/pose-gpu-runtime.json').read_text(encoding='utf-8'))
    target=root/'.runtime/ort-gpu-1.26.0'
    if metadata['version']!='1.26.0' or Path(metadata['target']).resolve()!=target.resolve() or not target.is_dir():
        raise RuntimeError('GPU runtime provenance does not match the installed isolated package')
    if 'onnxruntime' in sys.modules:
        loaded=sys.modules['onnxruntime']
        if not Path(loaded.__file__).resolve().is_relative_to(target.resolve()):
            raise RuntimeError('CPU ONNX Runtime is already loaded; restart this process for GPU pose')
    else:sys.path.insert(0,str(target))
    import torch
    import onnxruntime as ort
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    ort.preload_dlls()
    if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDA pose provider is unavailable')
    return dict(runtime=ort.__version__,path=str(target),wheel_sha256=metadata['sha256'])
