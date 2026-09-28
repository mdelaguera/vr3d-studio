"""VR 3D conversion engine: 2D photos/videos -> stereo 3D (SBS, VR180, anaglyph)."""

from .depth import CATALOG, DEFAULT_MODEL, ModelInfo, device_label, load_model
from .pipeline import Converter, Progress, compose
from .settings import (IMAGE_EXTS, SUFFIXES, VIDEO_EXTS, OutputFormat, OutputSettings, StereoSettings,
                       is_image, is_video, output_path_for)

__all__ = [
    "CATALOG", "DEFAULT_MODEL", "ModelInfo", "device_label", "load_model",
    "Converter", "Progress", "compose",
    "IMAGE_EXTS", "SUFFIXES", "VIDEO_EXTS", "OutputFormat", "OutputSettings", "StereoSettings",
    "is_image", "is_video", "output_path_for",
]
