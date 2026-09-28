"""DPT-style image preprocessing shared by the PyTorch and ONNX backends (cv2 is much faster than PIL)."""

import cv2
import numpy as np


class DPTPreprocessor:
    def __init__(self, config: dict):
        """`config` is a Hugging Face preprocessor_config.json dict (DPTImageProcessor)."""
        self.target = (config["size"]["height"], config["size"]["width"])
        self.keep_aspect = config.get("keep_aspect_ratio", False)
        self.multiple = config.get("ensure_multiple_of") or 1
        self.mean = np.array(config["image_mean"], dtype=np.float32)
        self.std = np.array(config["image_std"], dtype=np.float32)

    def input_size(self, h: int, w: int):
        th, tw = self.target
        if self.keep_aspect:
            # Same rule as the HF DPT processor: scale by whichever factor changes the image least
            sh, sw = th / h, tw / w
            s = sh if abs(1 - sh) < abs(1 - sw) else sw
            th, tw = h * s, w * s
        m = self.multiple
        return max(m, int(round(th / m) * m)), max(m, int(round(tw / m) * m))

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 HxWx3 -> normalized float32 1x3xhxw."""
        ih, iw = self.input_size(*bgr.shape[:2])
        rgb = cv2.cvtColor(cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
