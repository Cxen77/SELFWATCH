"""RTMW ONNX refinement on YOLO person boxes; MMPose COCO133 layout."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import time


class WholeBody:
    def __init__(self, model_path, device="cpu", max_people=6):
        from rtmlib import RTMPose
        import onnxruntime as ort
        path = Path(model_path).resolve()
        if not path.is_file() or max_people < 1:
            raise ValueError("Local ONNX model and positive max_people required")
        ort.set_default_logger_severity(3)
        self.model = RTMPose(str(path), model_input_size=(192, 256),
                            to_openpose=False, backend="onnxruntime", device='cpu')
        # The default ORT thread pool can oversubscribe a laptop alongside torch.
        # Replace its public inference session with a bounded CPU session.
        if device == "cpu":
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self.model.session = ort.InferenceSession(str(path), sess_options=options,
                                                      providers=["CPUExecutionProvider"])
        elif device.startswith('cuda'):
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            options.log_severity_level=3
            gpu_id=int(device.split(':')[1]) if ':' in device else 0
            self.model.session=ort.InferenceSession(str(path),sess_options=options,providers=[
                ('CUDAExecutionProvider',dict(device_id=gpu_id,gpu_mem_limit=2048*1024**2,
                    arena_extend_strategy='kNextPowerOfTwo',cudnn_conv_algo_search='HEURISTIC',
                    cudnn_conv_use_max_workspace='0',use_tf32='0')),'CPUExecutionProvider'])
        providers = self.model.session.get_providers()
        if device.startswith("cuda") and "CUDAExecutionProvider" not in providers:
            raise RuntimeError("CUDA pose requested but unavailable; explicitly select CPU or install GPU ONNX runtime")
        self.provenance = dict(checkpoint=str(path), sha256=sha256(path.read_bytes()).hexdigest(),
            schema="coco-wholebody133", input_wh=[192, 256], providers=providers,
            runtime=ort.__version__, max_people=max_people,
            cpu_intra_threads=2 if device == "cpu" else None,
            gpu_arena_limit_mb=2048 if device.startswith('cuda') else None,
            scores="Raw model landmark scores; not calibrated visibility probabilities")
        self.max_people = max_people

    def refine(self, observation, image):
        import numpy as np
        people = sorted((i for i, person in enumerate(observation.instances) if person.label.lower() == "person"),
                        key=lambda i: observation.instances[i].score, reverse=True)[:self.max_people]
        start = time.perf_counter()
        instances = list(observation.instances)
        if people:
            boxes = [instances[i].box_xyxy for i in people]
            points, scores = self.model(image, bboxes=boxes)
            if points.shape != (len(people), 133, 2) or scores.shape != (len(people), 133):
                raise ValueError(f"Unexpected RTMW topology: {points.shape}, {scores.shape}")
            if not np.isfinite(points).all() or not np.isfinite(scores).all():
                raise ValueError("Nonfinite RTMW output")
            for number, index in enumerate(people):
                xyz = np.column_stack((points[number], scores[number]))
                instances[index] = replace(instances[index], keypoints_xyc=tuple(tuple(map(float, p)) for p in xyz))
        return replace(observation, instances=tuple(instances), keypoint_schema="coco-wholebody133",
                       task="segment+wholebody" if observation.task == "segment" else "detect+wholebody",
                       model=observation.model+"+"+self.provenance["sha256"],
                       inference_ms=observation.inference_ms+(time.perf_counter()-start)*1000)
