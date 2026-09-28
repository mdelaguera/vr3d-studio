"""Weighted fusion of two depth models: one for crisp edges, one for volume."""

import numpy as np

from . import DepthModel, ModelInfo, normalize


class HybridDepth:
    def __init__(self, info: ModelInfo, edges: DepthModel, volume: DepthModel, edge_weight: float = 0.6):
        self.info = info
        self.edges = edges
        self.volume = volume
        self.edge_weight = edge_weight

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        w = self.edge_weight
        fused = w * self.edges.estimate(bgr) + (1.0 - w) * self.volume.estimate(bgr)
        return normalize(fused)
