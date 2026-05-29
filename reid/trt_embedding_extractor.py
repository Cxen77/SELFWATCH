"""
TRT Embedding Extractor Backend (Dynamic Batch)
=================================================
Drop-in replacement for EmbeddingExtractor that uses a pre-built TensorRT
dynamic-batch FP16 engine for OSNet inference while preserving:

  - Identical extract_batch() interface and return format
  - L2-normalized (N, 512) float32 numpy output
  - Pinned memory + async DMA preprocessing (mirrors EmbeddingExtractor)
  - GPU-side normalization (no CPU round-trip)
  - Dedicated non-default CUDA stream
  - PyTorch fallback (auto-activated if engine missing or fails to load)
  - Same compute_similarity() and compute_distance() static methods

Architecture
============
The TRT engine wraps ONLY the OSNet neural-net forward pass and supports
dynamic batching (N=1 to 16 crops per call).

Data flow for N crops:
    N Crops (BGR numpy, 128×128)
        → Stacked into single (N,H,W,3) array
        → BGR→RGB + float32 scale (CPU numpy)
        → Pinned memory staging buffer (N slice)
        → Async DMA → GPU (non-blocking, single transfer)
        → ImageNet normalization (GPU, pre-computed constants, batched)
        → TRT forward pass (GPU, FP16, dynamic shape set to N)
        → L2 normalization (GPU, batched F.normalize)
        → float32 numpy (single GPU→CPU transfer)

Usage
=====
    from reid.trt_embedding_extractor import TRTEmbeddingExtractor

    reid = TRTEmbeddingExtractor(
        engine_path="models/osnet_x1_0_dyn_fp16.engine",
        device="cuda:0",
    )
    embs = reid.extract_batch(crops)   # (N, 512) float32, L2-normalized

Integration
===========
In app.py / engine / orchestration, replace:
    from reid import EmbeddingExtractor
    reid = EmbeddingExtractor(...)

With:
    from reid.trt_embedding_extractor import TRTEmbeddingExtractor
    reid = TRTEmbeddingExtractor(engine_path="models/osnet_x1_0_dyn_fp16.engine")

Rollback
========
If engine is missing or fails, TRTEmbeddingExtractor automatically
falls back to EmbeddingExtractor (PyTorch) with a warning.
Set fallback=False to raise instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ─── Default paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT    = Path(__file__).resolve().parent.parent
_DEFAULT_ENGINE  = _PROJECT_ROOT / "models" / "osnet_x1_0_dyn_fp16.engine"
_DEFAULT_WEIGHTS = _PROJECT_ROOT / "weights" / "osnet" / "osnet_x1_0_msmt17.pth"

# ─── Preprocessing constants (must match embedding_extractor.py) ──────────────
_INPUT_HEIGHT  = 128
_INPUT_WIDTH   = 128
_PIXEL_MEAN    = [0.485, 0.456, 0.406]
_PIXEL_STD     = [0.229, 0.224, 0.225]
_EMBEDDING_DIM = 512

# Matches Engine Phase 3 MAX_BATCH
_MAX_BATCH     = 16   


class TRTEmbeddingExtractor:
    """
    TensorRT FP16 embedding extractor for OSNet ReID with Dynamic Batch support.

    Drop-in replacement for EmbeddingExtractor. Implements the same public API:
      - extract_batch(crops) → (N, 512) float32 numpy, L2-normalized
      - compute_similarity(emb_a, emb_b) → float
      - compute_distance(emb_a, emb_b) → float
      - .embedding_dim property
      - .device property
    """

    def __init__(
        self,
        engine_path: Union[str, Path] = _DEFAULT_ENGINE,
        device:      str               = "cuda:0",
        fallback:    bool              = True,
        # Fallback kwargs — passed to EmbeddingExtractor if TRT fails
        fallback_weights: Optional[Union[str, Path]] = None,
        fallback_half:    bool = True,
    ):
        self._device_str  = device
        self._device      = torch.device(device)
        self._engine_path = Path(engine_path)
        self._trt_ok      = False
        self._fallback    = None

        # TRT runtime objects
        self._engine:  object = None  
        self._context: object = None  

        # IO tensor names
        self._input_name:  str = "images"
        self._output_name: str = "features"

        # Max-sized GPU buffers
        self._input_buf:  Optional[torch.Tensor] = None   # (16, 3, H, W) FP32
        self._output_buf: Optional[torch.Tensor] = None   # (16, 512) FP32

        # Pre-allocated pinned staging buffer (CPU)
        self._pinned_buf: Optional[torch.Tensor] = None   # (16, 3, H, W) FP32

        # GPU normalization constants
        self._gpu_mean: Optional[torch.Tensor] = None
        self._gpu_std:  Optional[torch.Tensor] = None

        self._stream: Optional[torch.cuda.Stream] = None

        self._trt_ok = self._load_engine()

        if not self._trt_ok:
            if fallback:
                print("[TRT-ReID] ⚠ TRT engine failed to load — falling back to PyTorch EmbeddingExtractor")
                from reid.embedding_extractor import EmbeddingExtractor
                self._fallback = EmbeddingExtractor(
                    weights_path=fallback_weights or _DEFAULT_WEIGHTS,
                    device=device,
                    half=fallback_half,
                )
            else:
                raise RuntimeError(f"TRT engine not available at {engine_path} and fallback=False")
        else:
            print(f"[TRT-ReID] ✓ TRT ReID backend ready (Dynamic Batch)")
            print(f"[TRT-ReID]   Engine : {engine_path}")
            print(f"[TRT-ReID]   Device : {device}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Engine loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_engine(self) -> bool:
        if not self._engine_path.exists():
            print(f"[TRT-ReID] Engine file not found: {self._engine_path}")
            return False
        try:
            import tensorrt as trt
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")
            runtime = trt.Runtime(TRT_LOGGER)

            with open(self._engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())

            if self._engine is None: return False

            self._input_name  = self._engine.get_tensor_name(0)
            self._output_name = self._engine.get_tensor_name(1)

            self._context = self._engine.create_execution_context()

            # The engine uses an optimization profile, so we must set a valid shape
            # before querying dependent shapes or allocating buffers.
            self._context.set_input_shape(self._input_name, (_MAX_BATCH, 3, _INPUT_HEIGHT, _INPUT_WIDTH))
            
            in_shape  = tuple(self._context.get_tensor_shape(self._input_name))
            out_shape = tuple(self._context.get_tensor_shape(self._output_name))
            
            assert in_shape  == (_MAX_BATCH, 3, _INPUT_HEIGHT, _INPUT_WIDTH)
            assert out_shape == (_MAX_BATCH, _EMBEDDING_DIM)

            # Pre-allocate GPU buffers matching MAX_BATCH
            self._input_buf  = torch.zeros(in_shape,  dtype=torch.float32, device=self._device)
            self._output_buf = torch.zeros(out_shape, dtype=torch.float32, device=self._device)
            self._context.set_tensor_address(self._input_name,  self._input_buf.data_ptr())
            self._context.set_tensor_address(self._output_name, self._output_buf.data_ptr())

            # Pinned staging buffer
            self._pinned_buf = torch.zeros(in_shape, dtype=torch.float32, pin_memory=True)

            self._gpu_mean = torch.tensor(_PIXEL_MEAN, dtype=torch.float32, device=self._device).view(1, 3, 1, 1)
            self._gpu_std  = torch.tensor(_PIXEL_STD,  dtype=torch.float32, device=self._device).view(1, 3, 1, 1)

            self._stream = torch.cuda.Stream(device=self._device)

            # Warmup
            print("[TRT-ReID] Warming up TRT engine …")
            with torch.inference_mode():
                self._context.set_input_shape(self._input_name, (1, 3, _INPUT_HEIGHT, _INPUT_WIDTH))
                for _ in range(3):
                    with torch.cuda.stream(self._stream):
                        self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
                self._stream.synchronize()
            return True

        except Exception as e:
            print(f"[TRT-ReID] Engine load error: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API — extract_batch
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

        if self._fallback is not None:
            return self._fallback.extract_batch(crops)

        n = len(crops)
        
        # If N exceeds MAX_BATCH, fall back to sequential batches (rare for typical use)
        if n > _MAX_BATCH:
            results = []
            for i in range(0, n, _MAX_BATCH):
                results.append(self.extract_batch(crops[i:i+_MAX_BATCH]))
            return np.vstack(results)

        # 1. Resize and stack crops
        resized = []
        for crop in crops:
            if crop.shape[:2] != (_INPUT_HEIGHT, _INPUT_WIDTH):
                crop = cv2.resize(crop, (_INPUT_WIDTH, _INPUT_HEIGHT))
            resized.append(crop)
            
        # (N, H, W, 3) BGR
        batch_np = np.stack(resized, axis=0)
        
        # (N, H, W, 3) RGB → float32
        rgb = np.ascontiguousarray(batch_np[:, :, :, ::-1]).astype(np.float32) / 255.0
        chw = rgb.transpose(0, 3, 1, 2)  # (N, 3, H, W)

        # 2. CPU → Pinned buffer
        self._pinned_buf[:n].copy_(torch.from_numpy(chw))

        with torch.cuda.stream(self._stream):
            # 3. Pinned → GPU input buffer
            gpu_input_slice = self._input_buf[:n]
            gpu_input_slice.copy_(self._pinned_buf[:n], non_blocking=True)
            
            # 4. In-place ImageNet normalization
            gpu_input_slice.sub_(self._gpu_mean).div_(self._gpu_std)

            # 5. TRT forward pass
            self._context.set_input_shape(self._input_name, (n, 3, _INPUT_HEIGHT, _INPUT_WIDTH))
            self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)

            # 6. L2 normalize output slice
            gpu_out_slice = self._output_buf[:n]
            emb_normalized = F.normalize(gpu_out_slice, p=2, dim=1)

        # 7. Single stream sync + GPU → CPU transfer
        self._stream.synchronize()
        return emb_normalized.float().cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────
    #  Similarity / distance
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return float(np.dot(emb_a, emb_b))

    @staticmethod
    def compute_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        diff = emb_a - emb_b
        return float(np.sqrt(np.dot(diff, diff)))

    # ──────────────────────────────────────────────────────────────────────────
    #  Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int: return _EMBEDDING_DIM
    @property
    def device(self) -> str: return self._device_str
    @property
    def is_trt(self) -> bool: return self._trt_ok

    def prepare_for_tensorrt(self): pass


if __name__ == "__main__":
    print("=" * 60)
    print("  TRTEmbeddingExtractor — Verification (Dynamic)")
    print("=" * 60)

    extractor = TRTEmbeddingExtractor(fallback=True)

    person_a = np.full((_INPUT_HEIGHT, _INPUT_WIDTH, 3), [50, 80, 200], dtype=np.uint8)
    person_b = np.full((_INPUT_HEIGHT, _INPUT_WIDTH, 3), [200, 50, 50], dtype=np.uint8)

    embs = extractor.extract_batch([person_a, person_b])
    print(f"\n[OK] Batch shape: {embs.shape} (should be (2, 512))")
    print(f"[OK] L2 norm    : {np.linalg.norm(embs[0]):.6f} (should be ~1.0)")
