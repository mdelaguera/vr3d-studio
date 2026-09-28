"""Frame-to-frame depth smoothing for video: removes shimmer without smearing motion."""

import cv2
import numpy as np

PROBE_SIZE = (160, 90)


class DepthSmoother:
    def __init__(self, base_alpha: float = 0.6, motion_gain: float = 6.0, cut_threshold: float = 30.0):
        """
        base_alpha: weight of the new depth where nothing moves (lower = steadier).
        motion_gain: how quickly moving pixels switch to the new depth.
        cut_threshold: mean gray-level change (0-255) that counts as a scene cut.
        """
        self.base_alpha = base_alpha
        self.motion_gain = motion_gain
        self.cut_threshold = cut_threshold
        self.reset()

    def reset(self):
        self._prev_probe = None
        self._prev_depth = None

    def __call__(self, bgr: np.ndarray, depth: np.ndarray) -> np.ndarray:
        probe = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), PROBE_SIZE, interpolation=cv2.INTER_AREA)
        prev_probe, prev_depth = self._prev_probe, self._prev_depth
        self._prev_probe = probe

        if prev_depth is None or prev_depth.shape != depth.shape:
            self._prev_depth = depth
            return depth
        motion = cv2.absdiff(probe, prev_probe).astype(np.float32)
        if motion.mean() > self.cut_threshold:  # scene cut: start fresh, no blending across cuts
            self._prev_depth = depth
            return depth

        # Per-pixel alpha: static areas smooth heavily, moving areas follow the new depth
        alpha = np.clip(self.base_alpha + motion / 255.0 * self.motion_gain, self.base_alpha, 1.0)
        alpha = cv2.resize(alpha, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = alpha * depth + (1.0 - alpha) * prev_depth
        self._prev_depth = out
        return out
