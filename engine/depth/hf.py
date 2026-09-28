"""Transformers backend for DPT-family depth models (Depth Anything V2, MiDaS DPT)."""

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from . import ModelInfo, normalize
from .preprocess import DPTPreprocessor


class TransformersDepth:
    def __init__(self, info: ModelInfo, device: str):
        self.info = info
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.pre = DPTPreprocessor(AutoImageProcessor.from_pretrained(info.repo).to_dict())
        self.model = AutoModelForDepthEstimation.from_pretrained(info.repo, torch_dtype=self.dtype)
        self.model.to(device).eval()

    @torch.inference_mode()
    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        x = torch.from_numpy(self.pre(bgr)).to(self.device, self.dtype)
        pred = self.model(pixel_values=x).predicted_depth  # (1, ih, iw), relative inverse depth: high = near
        depth = normalize(pred[0].float().cpu().numpy())
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
