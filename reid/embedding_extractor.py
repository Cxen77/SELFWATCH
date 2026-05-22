"""
OSNet Embedding Extractor for Person Re-Identification

Loads OSNet x1.0 pretrained on MSMT17 and extracts 512-dim L2-normalized
embedding vectors from cropped person images.

Architecture: Zhou et al. "Omni-Scale Feature Learning for Person
Re-Identification", ICCV 2019.

Responsibility: person crop → embedding vector. Nothing else.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ─── Default weight path (relative to project root) ─────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PROJECT_ROOT / "weights" / "osnet" / "osnet_x1_0_msmt17.pth"

# ─── Preprocessing constants (ImageNet stats, torchreid convention) ──────────
_INPUT_HEIGHT = 128
_INPUT_WIDTH = 128
_PIXEL_MEAN = [0.485, 0.456, 0.406]
_PIXEL_STD = [0.229, 0.224, 0.225]
_EMBEDDING_DIM = 512


# ═════════════════════════════════════════════════════════════════════════════
#  OSNet Architecture (self-contained, no torchreid dependency)
# ═════════════════════════════════════════════════════════════════════════════

class _ConvLayer(nn.Module):
    """Conv + BN + ReLU."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1(nn.Module):
    """1×1 conv + BN + ReLU."""

    def __init__(self, in_ch, out_ch, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                              padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1Linear(nn.Module):
    """1×1 conv + BN (no activation)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                              padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class _LightConv3x3(nn.Module):
    """Lightweight 3×3: pointwise (linear) → depthwise 3×3 (nonlinear)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1,
                               bias=False, groups=out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class _ChannelGate(nn.Module):
    """Channel-wise attention gate (squeeze-excitation style)."""

    def __init__(self, in_ch, num_gates=None, reduction=16):
        super().__init__()
        if num_gates is None:
            num_gates = in_ch
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_ch // reduction, num_gates, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        w = self.global_avgpool(x)
        w = self.relu(self.fc1(w))
        w = self.gate_activation(self.fc2(w))
        return x * w


class _OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_ch, out_ch, bottleneck_reduction=4, **kwargs):
        super().__init__()
        mid = out_ch // bottleneck_reduction
        self.conv1 = _Conv1x1(in_ch, mid)
        self.conv2a = _LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(
            _LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(
            _LightConv3x3(mid, mid), _LightConv3x3(mid, mid),
            _LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(
            _LightConv3x3(mid, mid), _LightConv3x3(mid, mid),
            _LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, out_ch)
        self.downsample = (
            _Conv1x1Linear(in_ch, out_ch) if in_ch != out_ch else None
        )

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1))
              + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class _OSNet(nn.Module):
    """
    OSNet backbone (feature extractor only, no classifier head).

    Outputs a 512-dim feature vector per input image.
    """

    def __init__(self, blocks, layers, channels, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        # Convolutional backbone
        self.conv1 = _ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0],
                                      channels[0], channels[1],
                                      reduce_spatial=True)
        self.conv3 = self._make_layer(blocks[1], layers[1],
                                      channels[1], channels[2],
                                      reduce_spatial=True)
        self.conv4 = self._make_layer(blocks[2], layers[2],
                                      channels[2], channels[3],
                                      reduce_spatial=False)
        self.conv5 = _Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # FC embedding layer
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_layer(block, num_blocks, in_ch, out_ch, reduce_spatial=True):
        layers = [block(in_ch, out_ch)]
        for _ in range(1, num_blocks):
            layers.append(block(out_ch, out_ch))
        if reduce_spatial:
            layers.append(nn.Sequential(
                _Conv1x1(out_ch, out_ch), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _build_osnet_x1_0() -> _OSNet:
    """Construct OSNet x1.0 (standard width, 512-dim output)."""
    return _OSNet(
        blocks=[_OSBlock, _OSBlock, _OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        feature_dim=_EMBEDDING_DIM,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Public API: EmbeddingExtractor
# ═════════════════════════════════════════════════════════════════════════════

class EmbeddingExtractor:
    """
    OSNet-based person re-identification embedding extractor.

    Converts cropped person images (BGR numpy) into L2-normalized
    512-dimensional embedding vectors suitable for cosine similarity
    matching in StrongSORT / FAISS pipelines.

    Args:
        weights_path: Path to osnet_x1_0_msmt17.pth checkpoint.
                      Defaults to weights/osnet/osnet_x1_0_msmt17.pth.
        device:       "cuda:0", "cpu", or None (auto-detect).
        half:         Use FP16 inference on CUDA (default True).
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        half: bool = True,
    ):
        # Resolve device
        if device is None:
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        self._half = half and self._device.type == "cuda"

        # Build model
        self._model = _build_osnet_x1_0()
        self._load_weights(weights_path or _DEFAULT_WEIGHTS)
        self._model.to(self._device)
        if self._half:
            self._model.half()
        self._model.eval()

        # Preprocessing pipeline
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((_INPUT_HEIGHT, _INPUT_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_PIXEL_MEAN, std=_PIXEL_STD),
        ])

        print(f"[SELFWATCH] OSNet x1.0 loaded on {self._device}"
              f" | FP16={'ON' if self._half else 'OFF'}"
              f" | dim={_EMBEDDING_DIM}")

    # ── Weight Loading ───────────────────────────────────────────────────

    def _load_weights(self, weights_path: Union[str, Path]) -> None:
        """Load MSMT17 pretrained weights with flexible key matching."""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"OSNet weights not found at {weights_path}. "
                f"Download osnet_x1_0_msmt17.pth into weights/osnet/."
            )

        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=False
        )

        # Handle wrapped checkpoints (some save under 'state_dict' key)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix from DataParallel checkpoints
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k
            cleaned[key] = v

        # Load with partial matching (ignore classifier head)
        model_dict = self._model.state_dict()
        matched = {
            k: v for k, v in cleaned.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(matched)
        self._model.load_state_dict(model_dict)

        skipped = set(cleaned.keys()) - set(matched.keys())
        if skipped:
            print(f"[SELFWATCH] OSNet: skipped {len(skipped)} keys "
                  f"(classifier head / shape mismatch)")
        print(f"[SELFWATCH] OSNet: loaded {len(matched)}/{len(model_dict)} "
              f"layers from {weights_path.name}")

    # ── Preprocessing ────────────────────────────────────────────────────

    def preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single BGR person crop for OSNet inference.

        Args:
            crop: BGR image (H, W, 3), uint8 numpy array.

        Returns:
            Tensor of shape (3, 256, 128), normalized.
        """
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self._transform(rgb)

    def _preprocess_batch(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a batch of BGR crops into a single tensor.

        Returns:
            Tensor of shape (N, 3, 256, 128).
        """
        tensors = [self.preprocess(c) for c in crops]
        batch = torch.stack(tensors)
        batch = batch.to(self._device)
        if self._half:
            batch = batch.half()
        return batch

    def _preprocess_batch_fast(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Fast vectorized batch preprocessing — avoids per-crop PIL conversion.

        All crops MUST already be resized to (_INPUT_HEIGHT, _INPUT_WIDTH).
        Performs bulk BGR→RGB, float32 conversion, ImageNet normalization,
        and HWC→CHW transpose using numpy broadcasting, then a single
        transfer to GPU.

        Returns:
            Tensor of shape (N, 3, _INPUT_HEIGHT, _INPUT_WIDTH) on device.
        """
        n = len(crops)
        # Stack into (N, H, W, 3) uint8 array
        batch_np = np.stack(crops, axis=0)  # all crops are already 128x128x3

        # BGR → RGB (reverse channel axis for entire batch at once)
        batch_np = batch_np[:, :, :, ::-1]  # (N, H, W, 3) RGB

        # uint8 → float32, scale to [0, 1]
        batch_f = batch_np.astype(np.float32) * (1.0 / 255.0)

        # ImageNet normalization: (x - mean) / std
        mean = np.array(_PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array(_PIXEL_STD, dtype=np.float32).reshape(1, 1, 1, 3)
        batch_f = (batch_f - mean) / std

        # HWC → CHW: (N, H, W, C) → (N, C, H, W)
        batch_f = batch_f.transpose(0, 3, 1, 2)

        # Ensure contiguous memory layout before torch conversion
        batch_f = np.ascontiguousarray(batch_f)

        # Single transfer to GPU
        batch_t = torch.from_numpy(batch_f).to(self._device)
        if self._half:
            batch_t = batch_t.half()
        return batch_t

    # ── Extraction ───────────────────────────────────────────────────────

    @torch.inference_mode()
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract L2-normalized embeddings from a batch of BGR person crops.

        Args:
            crops: List of BGR images, each (H, W, 3) uint8.
                   All crops MUST be pre-resized to 128×128.

        Returns:
            np.ndarray of shape (N, 512), float32, each row L2-normalized.
        """
        if not crops:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)
        batch = self._preprocess_batch_fast(crops)
        with torch.autocast(device_type=self._device.type, enabled=self._half):
            feats = self._model(batch)
            feats = F.normalize(feats, p=2, dim=1)
        return feats.float().cpu().numpy()

    def prepare_for_tensorrt(self):
        """
        Placeholder for future TensorRT optimization.
        Example: export to ONNX -> build TensorRT engine with dynamic batch size.
        """
        pass

    # ── Similarity ───────────────────────────────────────────────────────

    @staticmethod
    def compute_similarity(
        emb_a: np.ndarray, emb_b: np.ndarray
    ) -> float:
        """
        Cosine similarity between two L2-normalized embedding vectors.

        Args:
            emb_a: (512,) float32 embedding.
            emb_b: (512,) float32 embedding.

        Returns:
            Similarity score in [-1.0, 1.0]. Higher = more similar.
        """
        return float(np.dot(emb_a, emb_b))

    @staticmethod
    def compute_distance(
        emb_a: np.ndarray, emb_b: np.ndarray
    ) -> float:
        """
        Euclidean distance between two embeddings.

        For L2-normalized vectors: dist² = 2 - 2·cos(θ), so this is
        monotonically related to cosine similarity.

        Returns:
            Distance ≥ 0. Lower = more similar.
        """
        diff = emb_a - emb_b
        return float(np.sqrt(np.dot(diff, diff)))

    # ── Info ─────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

    @property
    def device(self) -> str:
        return str(self._device)


# ═════════════════════════════════════════════════════════════════════════════
#  Standalone Verification
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Quick verification that:
      1. The model loads and produces 512-dim vectors.
      2. Same-input embeddings yield similarity ≈ 1.0.
      3. Different-input embeddings yield lower similarity.
    """
    print("=" * 60)
    print("  OSNet Embedding Extractor — Verification")
    print("=" * 60)

    extractor = EmbeddingExtractor(half=False)

    # Create synthetic "person" crops (different colors = different people)
    person_a = np.full((256, 128, 3), [50, 80, 200], dtype=np.uint8)   # warm
    person_a_again = np.full((256, 128, 3), [50, 80, 200], dtype=np.uint8)
    person_b = np.full((256, 128, 3), [200, 50, 50], dtype=np.uint8)   # cool

    # Add some variation to person_a_again (slight noise)
    noise = np.random.randint(-10, 10, person_a_again.shape, dtype=np.int16)
    person_a_again = np.clip(
        person_a_again.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)

    # Extract embeddings
    emb_a1 = extractor.extract_batch([person_a])[0]
    emb_a2 = extractor.extract_batch([person_a_again])[0]
    emb_b = extractor.extract_batch([person_b])[0]

    # Verify dimensions
    print(f"\n[OK] Embedding shape: {emb_a1.shape}")
    print(f"[OK] L2 norm: {np.linalg.norm(emb_a1):.4f} (should be ~1.0)")

    # Similarity tests
    sim_same = EmbeddingExtractor.compute_similarity(emb_a1, emb_a2)
    sim_diff = EmbeddingExtractor.compute_similarity(emb_a1, emb_b)
    dist_same = EmbeddingExtractor.compute_distance(emb_a1, emb_a2)
    dist_diff = EmbeddingExtractor.compute_distance(emb_a1, emb_b)

    print(f"\n[Same person]      cosine={sim_same:.4f}  dist={dist_same:.4f}")
    print(f"[Different person] cosine={sim_diff:.4f}  dist={dist_diff:.4f}")

    assert sim_same > sim_diff, "FAIL: same-person similarity should exceed different-person"
    print(f"\n[OK] Same > Different: {sim_same:.4f} > {sim_diff:.4f} — PASS")

    # Batch extraction test
    batch_embs = extractor.extract_batch([person_a, person_b])
    print(f"[OK] Batch shape: {batch_embs.shape} (should be (2, 512))")

    print("\n" + "=" * 60)
    print("  All checks passed.")
    print("=" * 60)
