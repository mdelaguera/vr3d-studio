"""Diffusers backend for Marigold LCM (few-step diffusion depth)."""

import cv2
import numpy as np
import torch
from PIL import Image

from . import ModelInfo, normalize


class MarigoldDepth:
    def __init__(self, info: ModelInfo, device: str, steps: int = 4):
        from diffusers import MarigoldDepthPipeline  # heavy import, only when this model is chosen
        self.info = info
        self.steps = steps
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = MarigoldDepthPipeline.from_pretrained(info.repo, torch_dtype=dtype).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pred = np.squeeze(self.pipe(rgb, num_inference_steps=self.steps).prediction)  # affine depth: high = far
        depth = normalize(pred.astype(np.float32), far_is_high=True)
        return depth if depth.shape == (h, w) else cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
