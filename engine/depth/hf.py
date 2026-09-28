"""Transformers backend for DPT-family depth models (Depth Anything V2, MiDaS DPT)."""

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from . import ModelInfo, normalize


class TransformersDepth:
    def __init__(self, info: ModelInfo, device: str):
        self.info = info
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        proc = AutoImageProcessor.from_pretrained(info.repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(info.repo, torch_dtype=self.dtype)
        self.model.to(device).eval()

        # Preprocess ourselves with cv2 on the full frame: much faster than the PIL-based processor
        self.target = (proc.size["height"], proc.size["width"])
        self.keep_aspect = getattr(proc, "keep_aspect_ratio", False)
        self.multiple = getattr(proc, "ensure_multiple_of", 1) or 1
        self.mean = np.array(proc.image_mean, dtype=np.float32)
        self.std = np.array(proc.image_std, dtype=np.float32)

    def _input_size(self, h: int, w: int):
        th, tw = self.target
        if self.keep_aspect:
            # Same rule as the HF DPT processor: scale by whichever factor changes the image least
            sh, sw = th / h, tw / w
            s = sh if abs(1 - sh) < abs(1 - sw) else sw
            th, tw = h * s, w * s
        m = self.multiple
        return max(m, int(round(th / m) * m)), max(m, int(round(tw / m) * m))

    @torch.inference_mode()
    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        ih, iw = self._input_size(h, w)
        rgb = cv2.cvtColor(cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device, self.dtype)

        pred = self.model(pixel_values=x).predicted_depth  # (1, ih, iw), relative inverse depth: high = near
        depth = normalize(pred[0].float().cpu().numpy())
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
