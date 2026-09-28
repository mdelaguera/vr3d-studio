"""
Depth models: a catalog with licensing metadata, and a single loader.

Contract for every model: estimate(bgr uint8 HxWx3) -> float32 HxW in [0, 1], 1 = nearest.
Only this package imports torch, so the rest of the engine stays lightweight.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DepthModel(Protocol):
    info: "ModelInfo"

    def estimate(self, bgr: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    repo: str            # Hugging Face repo (weights download on first use, then cached)
    license: str
    commercial_ok: bool
    download_mb: int
    speed: str           # "fast" | "medium" | "slow" (relative, per frame)
    blurb: str
    onnx: str = ""       # "repo:file" ONNX export, used for DirectML GPUs (AMD / Intel / any DX12 on Windows)


CATALOG = {
    m.id: m for m in [
        ModelInfo("da2-small", "Depth Anything V2 Small", "depth-anything/Depth-Anything-V2-Small-hf",
                  "Apache-2.0", True, 100, "fast", "Fast and sharp. Great default for video.",
                  onnx="onnx-community/depth-anything-v2-small:onnx/model.onnx"),
        ModelInfo("dpt-hybrid", "DPT-Hybrid (MiDaS)", "Intel/dpt-hybrid-midas",
                  "Apache-2.0", True, 490, "fast", "Classic, smooth depth. Good for landscapes."),
        ModelInfo("marigold-lcm", "Marigold LCM", "prs-eth/marigold-depth-lcm-v1-0",
                  "Apache-2.0", True, 1700, "slow", "Diffusion depth with rich volume. Best for photos."),
        ModelInfo("hybrid", "Hybrid (Depth Anything + Marigold)", "",
                  "Apache-2.0", True, 1800, "slow", "Clean edges plus organic volume. Highest quality."),
    ]
}
DEFAULT_MODEL = "da2-small"


def directml_available() -> bool:
    try:
        import onnxruntime
        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        return False


def pick_device() -> str:
    """cuda | mps | dml | cpu. `dml` = PyTorch has no GPU, but ONNX Runtime can use one via DirectML."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "dml" if directml_available() else "cpu"


def device_label() -> str:
    """Human-readable description of where models will run."""
    device = pick_device()
    if device == "cuda":
        import torch
        return f"{torch.cuda.get_device_name(0)} · CUDA"
    return {"mps": "Apple GPU · Metal", "dml": "GPU · DirectML"}.get(device, "CPU (no GPU acceleration)")


def load_model(model_id: str = DEFAULT_MODEL, device: str = None) -> DepthModel:
    info = CATALOG.get(model_id)
    if info is None:
        raise ValueError(f"Unknown depth model '{model_id}'. Choose from: {', '.join(CATALOG)}")
    device = device or pick_device()
    if model_id == "hybrid":
        from .hybrid import HybridDepth
        return HybridDepth(info, load_model("da2-small", device), load_model("marigold-lcm", device))
    if device == "dml":
        if info.onnx:
            from .onnx import OnnxDepth
            return OnnxDepth(info)
        device = "cpu"  # no ONNX export for this model: PyTorch on CPU
    if model_id == "marigold-lcm":
        from .marigold import MarigoldDepth
        return MarigoldDepth(info, device)
    from .hf import TransformersDepth
    return TransformersDepth(info, device)


def normalize(raw: np.ndarray, far_is_high: bool = False) -> np.ndarray:
    """Robust [0, 1] scaling (1st-99th percentile), 1 = nearest."""
    small = raw[::4, ::4] if min(raw.shape) >= 64 else raw
    lo, hi = np.percentile(small, (1, 99))
    out = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)
    return 1.0 - out if far_is_high else out
