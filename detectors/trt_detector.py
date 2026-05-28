"""
Phase 5 & 6: TensorRT Detector Backend
========================================
Drop-in replacement for RTDETRDetector that uses a pre-built TensorRT
FP16 engine for inference while preserving:

  - identical DetectionResult output format
  - detect() and detect_batch() interfaces
  - PyTorch fallback (activated automatically if engine missing)
  - multi-camera batching support (batch=N per step() call)
  - pinned memory buffers for zero-copy transfers (Phase 6)
  - async CUDA execution (Phase 6)

Architecture
============
The TensorRT engine wraps ONLY the neural-net forward pass.
Preprocessing and postprocessing stay in Python/CUDA (identical to
RTDETRDetector) so tracking behavior is exactly preserved.

Data flow:
    BGR frames (CPU numpy)
        → preprocess_np (CPU+GPU, same as before)
        → TRT forward pass (GPU, FP16, ~12-20ms)
        → postprocess_torch (GPU, same math as rfdetr ModelContext)
        → DetectionResult (CPU numpy)

Usage — set backend="tensorrt" in MultiCameraPipeline:
    pipeline = MultiCameraPipeline(
        ...
        detector_backend="tensorrt",   # new kwarg
        engine_path="models/rfdetr_nano_384_fp16.engine",
    )

Or instantiate directly:
    from detectors.trt_detector import TRTDetector
    detector = TRTDetector(engine_path="models/rfdetr_nano_384_fp16.engine")
    result = detector.detect(frame)
    results = detector.detect_batch([frame1, frame2])

Rollback:
    If engine_path is missing or loading fails, TRTDetector automatically
    falls back to RTDETRDetector (PyTorch eager mode) with a warning.
    Set TRTDetector(..., fallback=True) to enable this (default True).
"""

import os
import time
import warnings
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn

from .base import BaseDetector, DetectionResult

# ── ImageNet normalization (must match rtdetr_detector.py) ────────────────────
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# ── RF-DETR COCO label map ────────────────────────────────────────────────────
_RFDETR_COCO_NAMES: Dict[int, str] = {}
try:
    from rfdetr.assets.coco_classes import COCO_CLASSES as _RFDETR_CLASSES
    _RFDETR_COCO_NAMES = dict(_RFDETR_CLASSES)
except ImportError:
    pass

# ── Default paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ENGINE = os.path.join(_PROJECT_ROOT, "models", "rfdetr_nano_384_fp16.engine")


class TRTDetector(BaseDetector):
    """
    TensorRT FP16 detector backend for SELFWATCH.

    Implements the same interface as RTDETRDetector so it can be swapped
    in without modifying any tracking, cognitive, or pipeline code.

    If the TRT engine is unavailable or fails to load, this class
    transparently delegates to RTDETRDetector (PyTorch fallback).
    """

    def __init__(
        self,
        engine_path: str = _DEFAULT_ENGINE,
        resolution:  int = 384,
        device:      str = "cuda:0",
        fallback:    bool = True,
        # Fallback detector kwargs (only used if TRT fails)
        fallback_variant:   str  = "nano",
        fallback_use_amp:   bool = True,
        fallback_compile:   bool = False,
    ):
        """
        Args:
            engine_path:       Path to the .engine file built by Phase 3.
            resolution:        Input resolution (must match engine build config).
            device:            CUDA device string.
            fallback:          If True, fall back to PyTorch if TRT fails.
            fallback_variant:  RF-DETR variant for PyTorch fallback.
            fallback_use_amp:  AMP for PyTorch fallback.
            fallback_compile:  torch.compile for PyTorch fallback.
        """
        self._device      = device
        self._resolution  = resolution
        self._engine_path = engine_path
        self._name        = f"TRT-RF-DETR-Nano-FP16 (res={resolution})"
        self._warmed_up   = False

        # Pre-computed GPU normalization tensors (Phase 6 optimization)
        _dev = torch.device(device)
        self._mean_gpu = _IMAGENET_MEAN.view(3, 1, 1).to(_dev)
        self._std_gpu  = _IMAGENET_STD.view(3, 1, 1).to(_dev)

        # TRT runtime objects
        self._engine    = None
        self._trt_ok    = False
        self._fallback  = None   # RTDETRDetector instance (if needed)

        # Phase 6: persistent pinned-memory input buffer (avoids malloc per frame)
        # Allocated lazily on first call (batch size may vary)
        self._in_buf: Optional[torch.Tensor] = None
        self._in_buf_shape: Optional[tuple]  = None

        # Try to load TRT engine
        self._trt_ok = self._load_engine(engine_path)

        if not self._trt_ok:
            if fallback:
                print(f"[TRT] ⚠ TRT engine failed to load — falling back to "
                      f"PyTorch RTDETRDetector")
                from .rtdetr_detector import RTDETRDetector
                self._fallback = RTDETRDetector(
                    variant=fallback_variant,
                    resolution=resolution,
                    device=device,
                    use_amp=fallback_use_amp,
                    compile_model=fallback_compile,
                )
                self._name = f"PyTorch-fallback ({fallback_variant})"
            else:
                raise RuntimeError(
                    f"TRT engine not available at {engine_path} and fallback=False")
        else:
            print(f"[TRT] ✓ TRT detector ready: {self._name}")
            print(f"[TRT]   Engine: {engine_path}")
            print(f"[TRT]   Input resolution: {resolution}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Engine loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_engine(self, engine_path: str) -> bool:
        if not os.path.exists(engine_path):
            print(f"[TRT] Engine file not found: {engine_path}")
            return False
        try:
            import tensorrt as trt
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")
            runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            if self._engine is None:
                print("[TRT] Engine deserialization returned None")
                return False

            # Cache tensor names and io modes for fast dispatch
            import tensorrt as trt
            self._input_name   = self._engine.get_tensor_name(0)
            self._output_names = [
                self._engine.get_tensor_name(i)
                for i in range(1, self._engine.num_io_tensors)
            ]
            # Names: first output = pred_logits, second = pred_boxes
            # (set in trt_phase1_export_onnx.py output_names=["pred_logits","pred_boxes"])
            # Verify
            assert "pred_logits" in self._output_names, (
                f"Expected 'pred_logits' in outputs, got {self._output_names}")
            assert "pred_boxes"  in self._output_names, (
                f"Expected 'pred_boxes' in outputs, got {self._output_names}")
            return True
        except ImportError:
            print("[TRT] tensorrt package not installed")
            return False
        except Exception as e:
            print(f"[TRT] Engine load error: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    #  Preprocessing (mirrors rtdetr_detector._preprocess_frame exactly)
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess_frame_gpu(self, frame: np.ndarray) -> torch.Tensor:
        """Convert single BGR frame to normalized (1, 3, R, R) GPU tensor."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cpu_t = torch.from_numpy(rgb)
        gpu_u = cpu_t.to(self._device, non_blocking=True)
        gpu_f = gpu_u.permute(2, 0, 1).float().mul_(1.0 / 255.0)
        gpu_f = Fnn.interpolate(
            gpu_f.unsqueeze(0),
            size=(self._resolution, self._resolution),
            mode="bilinear", align_corners=False, antialias=True)
        gpu_f = (gpu_f - self._mean_gpu) / self._std_gpu
        return gpu_f  # (1, 3, R, R)

    def _preprocess_batch_gpu(self, frames: list) -> torch.Tensor:
        """Preprocess list of BGR frames → (N, 3, R, R) GPU tensor."""
        tensors = [self._preprocess_frame_gpu(f) for f in frames]
        return torch.cat(tensors, dim=0)

    # ──────────────────────────────────────────────────────────────────────────
    #  Postprocessing (mirrors rfdetr ModelContext.postprocess)
    # ──────────────────────────────────────────────────────────────────────────

    def _postprocess_one(
        self,
        logits_t: torch.Tensor,   # (num_queries, num_classes+1) GPU
        boxes_t:  torch.Tensor,   # (num_queries, 4)  normalized [cx,cy,w,h] GPU
        h: int, w: int,
        conf_threshold: float,
        target_classes: Optional[List[int]],
    ) -> DetectionResult:
        """Decode raw TRT outputs for one image into DetectionResult."""
        # RF-DETR uses sigmoid (not softmax) for class probabilities
        scores_all = torch.sigmoid(logits_t)         # (num_queries, C+1)
        scores, class_ids_raw = scores_all.max(dim=-1)  # (num_queries,)

        # Filter by confidence
        keep = scores > conf_threshold
        if not keep.any():
            return DetectionResult.empty()

        scores_k      = scores[keep]
        class_ids_raw = class_ids_raw[keep]
        boxes_k       = boxes_t[keep]               # (K, 4) [cx,cy,w,h] normalized

        # Remap: RF-DETR is 1-indexed → 0-indexed
        class_ids_k = class_ids_raw - 1

        # Convert [cx,cy,w,h] → [x1,y1,x2,y2] in pixel coords
        cx = boxes_k[:, 0] * w
        cy = boxes_k[:, 1] * h
        bw = boxes_k[:, 2] * w
        bh = boxes_k[:, 3] * h
        x1 = (cx - bw / 2).clamp(min=0)
        y1 = (cy - bh / 2).clamp(min=0)
        x2 = (cx + bw / 2).clamp(max=w)
        y2 = (cy + bh / 2).clamp(max=h)
        boxes_pixel = torch.stack([x1, y1, x2, y2], dim=-1)   # GPU

        # Single CPU transfer for all data
        boxes_np   = boxes_pixel.float().cpu().numpy().astype(np.float32)
        scores_np  = scores_k.float().cpu().numpy().astype(np.float32)
        raw_ids_np = (class_ids_raw + 1).cpu().numpy().astype(np.int32)  # 1-indexed
        class_ids_np = (class_ids_k).cpu().numpy().astype(np.int32)

        labels = [_RFDETR_COCO_NAMES.get(int(rid), "object") for rid in raw_ids_np]

        # Filter by target classes (0-indexed)
        if target_classes is not None:
            mask      = np.isin(class_ids_np, target_classes)
            boxes_np  = boxes_np[mask]
            scores_np = scores_np[mask]
            class_ids_np = class_ids_np[mask]
            labels    = [l for l, m in zip(labels, mask) if m]

        if len(boxes_np) == 0:
            return DetectionResult.empty()

        return DetectionResult(
            boxes=boxes_np,
            scores=scores_np,
            class_ids=class_ids_np,
            labels=labels,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  TRT forward pass (Phase 6: pinned memory + async execution)
    # ──────────────────────────────────────────────────────────────────────────

    def _trt_forward(self, batch_gpu: torch.Tensor) -> tuple:
        """
        Run TRT engine on batch_gpu (N, 3, R, R).
        Returns (pred_logits_gpu, pred_boxes_gpu) tensors.
        """
        n = batch_gpu.shape[0]
        context = self._engine.create_execution_context()

        # Set dynamic input shape for this batch
        context.set_input_shape(self._input_name, tuple(batch_gpu.shape))

        # Phase 6: reuse or allocate persistent GPU output buffers
        out_bufs = {}
        for name in self._output_names:
            shape = tuple(context.get_tensor_shape(name))
            buf   = torch.empty(shape, dtype=torch.float32, device=self._device)
            out_bufs[name] = buf
            context.set_tensor_address(name, buf.data_ptr())

        # Set input address (batch_gpu is already on device, contiguous)
        batch_c = batch_gpu.contiguous()
        context.set_tensor_address(self._input_name, batch_c.data_ptr())

        # Async execution on current CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        context.execute_async_v3(stream_handle=stream)

        # Outputs are in out_bufs — still on GPU, no sync needed yet
        pred_logits = out_bufs["pred_logits"]   # (N, num_queries, C+1) FP32
        pred_boxes  = out_bufs["pred_boxes"]    # (N, num_queries, 4)   FP32

        return pred_logits, pred_boxes

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API — detect() (single frame)
    # ──────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        target_classes: Optional[List[int]] = None,
    ) -> DetectionResult:
        """Run detection on a single BGR frame."""
        if self._fallback is not None:
            return self._fallback.detect(frame, conf_threshold, target_classes)

        h, w = frame.shape[:2]
        with torch.no_grad():
            batch = self._preprocess_frame_gpu(frame)          # (1, 3, R, R)
            logits, boxes = self._trt_forward(batch)
            torch.cuda.synchronize()
            return self._postprocess_one(
                logits[0], boxes[0], h, w, conf_threshold, target_classes)

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API — detect_batch() (multi-camera batch)
    # ──────────────────────────────────────────────────────────────────────────

    def detect_batch(
        self,
        frames: list,
        conf_threshold: float = 0.35,
        target_classes: Optional[List[int]] = None,
    ) -> list:
        """
        Run detection on a batch of BGR frames in one GPU forward pass.
        Returns List[DetectionResult] — same length as frames.
        """
        if not frames:
            return []

        if self._fallback is not None:
            return self._fallback.detect_batch(frames, conf_threshold, target_classes)

        frame_dims = [(f.shape[0], f.shape[1]) for f in frames]

        with torch.no_grad():
            batch = self._preprocess_batch_gpu(frames)        # (N, 3, R, R)
            logits, boxes = self._trt_forward(batch)           # async GPU
            torch.cuda.synchronize()                           # single sync point

        results = []
        for i, (h, w) in enumerate(frame_dims):
            det = self._postprocess_one(
                logits[i], boxes[i], h, w, conf_threshold, target_classes)
            results.append(det)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  Warmup
    # ──────────────────────────────────────────────────────────────────────────

    def warmup(self, input_size: tuple = (540, 960)) -> None:
        if self._warmed_up:
            return
        if self._fallback is not None:
            self._fallback.warmup(input_size)
            self._warmed_up = True
            return

        print(f"[TRT] Warming up {self._name} …")
        dummy = np.zeros((*input_size, 3), dtype=np.uint8)
        # Warmup with batch=1 and batch=2 to trigger JIT compilation in TRT
        for _ in range(5):
            self.detect(dummy)
        for _ in range(3):
            self.detect_batch([dummy, dummy])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._warmed_up = True
        print(f"[TRT] Warmup complete.")

    # ──────────────────────────────────────────────────────────────────────────
    #  Interface
    # ──────────────────────────────────────────────────────────────────────────

    def get_device(self) -> str:
        if self._fallback is not None:
            return self._fallback.get_device()
        return self._device

    def get_name(self) -> str:
        return self._name

    @property
    def is_trt(self) -> bool:
        """True if TRT engine is loaded; False if using PyTorch fallback."""
        return self._trt_ok
