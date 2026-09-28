"""Rectilinear eye images -> VR180 side-by-side equirectangular frames."""

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=8)
def _remap_tables(src_w: int, src_h: int, eye_size: int, h_fov: float):
    """
    For each output pixel (longitude/latitude on the front hemisphere), the source pixel
    a pinhole camera with `h_fov` would see. Cached: built once per resolution + FOV.
    """
    t = (np.arange(eye_size, dtype=np.float32) + 0.5) / eye_size
    lon = ((t - 0.5) * np.pi)[None, :]           # -90°..+90° left to right
    lat = ((0.5 - t) * np.pi)[:, None]           # +90°..-90° top to bottom

    tan_h = np.tan(np.radians(h_fov) / 2.0)
    tan_v = tan_h * src_h / src_w                # vertical FOV follows the source aspect ratio
    # Ray (sin lon cos lat, sin lat, cos lon cos lat) projected onto the image plane z = 1
    x = np.tan(lon) / tan_h                      # in [-1, 1] inside the source frame
    y = np.tan(lat) / np.cos(lon) / tan_v
    inside = (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)

    map_x = np.where(inside, (x + 1.0) * 0.5 * (src_w - 1), -1.0).astype(np.float32)
    map_y = np.where(inside, (1.0 - y) * 0.5 * (src_h - 1), -1.0).astype(np.float32)
    # Convert to the fixed-point format cv2.remap reads fastest
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)


def project_eye(eye: np.ndarray, eye_size: int, h_fov: float) -> np.ndarray:
    h, w = eye.shape[:2]
    m1, m2 = _remap_tables(w, h, eye_size, float(h_fov))
    return cv2.remap(eye, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def vr180_sbs(left: np.ndarray, right: np.ndarray, eye_size: int = 1920, h_fov: float = 110.0) -> np.ndarray:
    return np.hstack((project_eye(left, eye_size, h_fov), project_eye(right, eye_size, h_fov)))
