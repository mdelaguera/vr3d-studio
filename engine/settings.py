"""Shared settings types. Every engine module speaks these + plain numpy arrays."""

import os
from dataclasses import dataclass
from enum import Enum


class OutputFormat(str, Enum):
    VR180 = "vr180"
    SBS_FULL = "sbs_full"
    SBS_HALF = "sbs_half"
    ANAGLYPH = "anaglyph"
    DEPTH = "depth_only"

    @property
    def is_stereo(self) -> bool:
        return self is not OutputFormat.DEPTH


# Filename suffixes VR players (Pico, Quest, Skybox, DeoVR) use to auto-detect the layout
SUFFIXES = {
    OutputFormat.VR180: "_180_SBS",
    OutputFormat.SBS_FULL: "_3DH_SBS",
    OutputFormat.SBS_HALF: "_3DH_Half_SBS",
    OutputFormat.ANAGLYPH: "_3D_Anaglyph",
    OutputFormat.DEPTH: "_Depth",
}

VIDEO_EXTS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm"})
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class StereoSettings:
    strength: float = 0.035    # max disparity as a fraction of image width
    focus: float = 0.5         # zero-parallax depth when auto_focus is off
    auto_focus: bool = True    # anchor zero parallax to the background
    swap_eyes: bool = False


@dataclass(frozen=True)
class OutputSettings:
    format: OutputFormat = OutputFormat.SBS_FULL
    vr_fov: float = 110.0      # horizontal FOV the source occupies inside the 180° dome
    vr_eye_size: int = 1920    # per-eye resolution of VR180 output


def output_path_for(src: str, fmt: OutputFormat, out_dir: str = None, ext: str = None) -> str:
    """`clip.mp4` -> `clip_180_SBS.mp4`, without doubling an existing suffix."""
    root, src_ext = os.path.splitext(os.path.basename(src))
    suffix = SUFFIXES[fmt]
    if not root.endswith(suffix):
        root += suffix
    return os.path.join(out_dir or os.path.dirname(src), root + (ext or src_ext))


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS
