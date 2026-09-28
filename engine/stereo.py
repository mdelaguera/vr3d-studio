"""
Depth-image-based rendering: one image + depth -> left/right eye views.

Two-layer approach: the foreground is warped over a background plate that has the
foreground removed and filled in, so disoccluded areas show background, not stretched
edges (no "ghost limbs"). Background disparity is anchored at zero, subjects pop forward.
"""

import cv2
import numpy as np

from .settings import StereoSettings

FG_MARGIN = 0.08     # depth above the anchor that counts as foreground
FG_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def fill_holes(img: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """
    Push-pull pyramid fill: holes take the weighted average of surrounding known pixels.
    ~50x faster than cv2.inpaint(TELEA) at 1080p; plenty for thin disocclusion slivers.
    """
    known = (~holes).astype(np.float32)
    pyramid = [(img.astype(np.float32) * known[..., None], known)]
    while min(pyramid[-1][1].shape) > 4:
        color, weight = pyramid[-1]
        size = ((weight.shape[1] + 1) // 2, (weight.shape[0] + 1) // 2)
        # INTER_AREA averages, so color stays premultiplied by weight
        pyramid.append((cv2.resize(color, size, interpolation=cv2.INTER_AREA),
                        cv2.resize(weight, size, interpolation=cv2.INTER_AREA)))

    color, weight = pyramid[-1]
    filled = color / np.maximum(weight, 1e-6)[..., None]
    for color, weight in reversed(pyramid[:-1]):
        up = cv2.resize(filled, (weight.shape[1], weight.shape[0]), interpolation=cv2.INTER_LINEAR)
        filled = color + up * (1.0 - weight[..., None])  # premultiplied "over": known pixels win
    return np.clip(filled, 0, 255).astype(np.uint8)


class StereoRenderer:
    def __init__(self):
        self._grid_shape = None
        self._grid = None

    def _pixel_grid(self, h: int, w: int):
        if self._grid_shape != (h, w):
            self._grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid_shape = (h, w)
        return self._grid

    def render(self, bgr: np.ndarray, depth: np.ndarray, settings: StereoSettings):
        """Returns (left, right) uint8 images the same size as `bgr`."""
        h, w = bgr.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = depth.astype(np.float32, copy=False)

        # Stretch to the scene's actual depth range (percentiles on a cheap downsample)
        small = depth[::4, ::4]
        lo, hi, p5 = np.percentile(small, (2, 98, 5))
        if hi - lo > 0.10:
            depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            p5 = (p5 - lo) / (hi - lo)
        anchor = float(p5) if settings.auto_focus else settings.focus

        disparity = np.maximum(depth - anchor, 0.0) * (w * settings.strength)

        fg = (depth > anchor + FG_MARGIN).astype(np.uint8)
        background = fill_holes(bgr, cv2.dilate(fg, FG_DILATE).astype(bool))
        alpha = cv2.GaussianBlur(fg.astype(np.float32), (5, 5), 0)

        grid_x, grid_y = self._pixel_grid(h, w)
        half = 0.5 * disparity
        left = self._eye(bgr, background, alpha, grid_x + half, grid_y)
        right = self._eye(bgr, background, alpha, grid_x - half, grid_y)
        return (right, left) if settings.swap_eyes else (left, right)

    @staticmethod
    def _eye(fg, background, alpha, map_x, map_y):
        warped = cv2.remap(fg, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.blendLinear(warped, background, a, 1.0 - a)


def side_by_side(left: np.ndarray, right: np.ndarray, half: bool = False) -> np.ndarray:
    if half:
        h, w = left.shape[:2]
        size = (w // 2, h)
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    return np.hstack((left, right))


def anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Red/cyan: red channel from the left eye, blue+green from the right (BGR order)."""
    return np.dstack((right[..., 0], right[..., 1], left[..., 2]))


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.clip(depth, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
