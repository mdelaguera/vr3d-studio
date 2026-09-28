"""
ONNX Runtime backend. With onnxruntime-directml it runs on any DirectX 12 GPU on Windows
(AMD, Intel, NVIDIA), which covers the machines PyTorch/CUDA can't accelerate.
"""

import json
from collections import OrderedDict

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from . import ModelInfo, normalize
from .preprocess import DPTPreprocessor

MAX_SESSIONS = 3   # one per input size in use (e.g. preview + video); each holds GPU memory


class OnnxDepth:
    def __init__(self, info: ModelInfo, use_gpu: bool = True):
        import onnxruntime as ort

        self.ort = ort
        self.info = info
        repo, filename = info.onnx.split(":")
        with open(hf_hub_download(repo, "preprocessor_config.json"), encoding="utf-8") as fh:
            self.pre = DPTPreprocessor(json.load(fh))
        self.model_path = hf_hub_download(repo, filename)
        self.on_gpu = use_gpu and "DmlExecutionProvider" in ort.get_available_providers()
        self._sessions = OrderedDict()
        probe = self._session(None)  # validates the model and reads its input signature
        self.input_name = probe.get_inputs()[0].name
        self._dims = probe.get_inputs()[0].shape   # e.g. ['batch_size', 3, 'height', 'width']

    def _session(self, shape):
        """
        DirectML is ~6x faster when every dimension is fixed, so on GPU we build one session per
        input size (free-dimension overrides) and keep the most recent few.
        """
        key = shape if self.on_gpu else None
        if key in self._sessions:
            self._sessions.move_to_end(key)
            return self._sessions[key]
        opts = self.ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if self.on_gpu:
            providers.insert(0, "DmlExecutionProvider")
            opts.enable_mem_pattern = False  # required by DirectML
            if shape is not None:
                for dim, value in zip(self._dims, shape):
                    if isinstance(dim, str):
                        opts.add_free_dimension_override_by_name(dim, value)
        session = self.ort.InferenceSession(self.model_path, opts, providers=providers)
        self._sessions[key] = session
        if len(self._sessions) > MAX_SESSIONS:
            self._sessions.popitem(last=False)
        return session

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        x = self.pre(bgr)
        pred = self._session(x.shape).run(None, {self.input_name: x})[0]   # (1, ih, iw): high = near
        depth = normalize(np.squeeze(pred).astype(np.float32))
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
