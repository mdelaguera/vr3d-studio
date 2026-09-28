"""
Engine tests. Fast by default (synthetic depth model, no downloads).
Run the real-model check with:  pytest -m model
"""

import os
import shutil
import threading

import cv2
import numpy as np
import pytest

from engine import (Converter, OutputFormat, OutputSettings, StereoSettings, output_path_for)
from engine import media
from engine.depth import ModelInfo, normalize
from engine.stereo import StereoRenderer, fill_holes
from engine.temporal import DepthSmoother
from engine.vr180 import vr180_sbs

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class BlobDepth:
    """Synthetic model: a near disc in the middle of a far background."""
    info = ModelInfo("fake", "Fake", "", "MIT", True, 0, "fast", "")

    def estimate(self, bgr):
        h, w = bgr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        disc = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(h, w) / 4) ** 2
        return np.where(disc, 0.9, 0.1).astype(np.float32)


def stripes(h=120, w=160):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, ::8] = 255
    img[..., 2] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    return img


def test_fill_holes_keeps_known_pixels_and_fills_holes_from_neighbours():
    img = np.full((64, 64, 3), 200, np.uint8)
    holes = np.zeros((64, 64), bool)
    holes[20:40, 20:40] = True
    img[holes] = 0
    out = fill_holes(img, holes)
    assert (out[~holes] == 200).all()
    assert abs(int(out[30, 30, 0]) - 200) <= 2


def test_background_stays_put_and_foreground_shifts_opposite_per_eye():
    img = stripes()
    depth = BlobDepth().estimate(img)
    left, right = StereoRenderer().render(img, depth, StereoSettings(strength=0.05))
    assert left.shape == right.shape == img.shape
    assert np.array_equal(left[:10], img[:10])            # far background: zero disparity
    centre = slice(55, 65)
    assert not np.array_equal(left[centre], right[centre])  # near subject: parallax


def test_swap_eyes_swaps():
    img, r = stripes(), StereoRenderer()
    depth = BlobDepth().estimate(img)
    l1, r1 = r.render(img, depth, StereoSettings())
    l2, r2 = r.render(img, depth, StereoSettings(swap_eyes=True))
    assert np.array_equal(l1, r2) and np.array_equal(r1, l2)


@pytest.mark.parametrize("fmt,shape", [
    (OutputFormat.SBS_FULL, (120, 320, 3)),
    (OutputFormat.SBS_HALF, (120, 160, 3)),
    (OutputFormat.ANAGLYPH, (120, 160, 3)),
    (OutputFormat.DEPTH, (120, 160, 3)),
    (OutputFormat.VR180, (64, 128, 3)),
])
def test_output_shapes(fmt, shape):
    out = Converter(BlobDepth()).render_frame(stripes(), StereoSettings(), OutputSettings(fmt, vr_eye_size=64))
    assert out.shape == shape and out.dtype == np.uint8


def test_vr180_centre_matches_source_and_corners_are_black():
    img = np.full((90, 160, 3), 255, np.uint8)
    out = vr180_sbs(img, img, eye_size=100, h_fov=110)
    assert out[50, 50].min() == 255      # straight ahead sees the image
    assert out[2, 2].max() == 0          # straight up/left is outside the camera FOV


def test_normalize_is_robust_to_outliers():
    raw = np.linspace(0, 1, 10000, dtype=np.float32).reshape(100, 100)
    raw[0, 0] = 1000.0
    out = normalize(raw)
    # Plain min/max scaling would squash this to ~0.0005 because of the single outlier
    assert out.min() == 0.0 and out.max() == 1.0 and 0.45 < out[50, 50] < 0.55


def test_smoother_blends_static_frames_and_resets_on_scene_cut():
    s = DepthSmoother(base_alpha=0.5)
    frame = np.zeros((90, 160, 3), np.uint8)
    s(frame, np.zeros((90, 160), np.float32))
    assert s(frame, np.ones((90, 160), np.float32)).mean() == pytest.approx(0.5)
    cut = np.full_like(frame, 255)
    assert s(cut, np.zeros((90, 160), np.float32)).mean() == 0.0


def test_output_naming_does_not_double_suffix():
    assert output_path_for("/v/clip.mp4", OutputFormat.VR180).endswith("clip_180_SBS.mp4")
    assert output_path_for("/v/clip_180_SBS.mp4", OutputFormat.VR180).endswith("clip_180_SBS.mp4")


def test_image_roundtrip_with_unicode_path(tmp_path):
    src = tmp_path / "fénykép ü.png"
    media.write_image(str(src), stripes())
    dst = tmp_path / "out.jpg"
    Converter(BlobDepth()).convert_image(str(src), str(dst), StereoSettings(), OutputSettings())
    assert media.read_image(str(dst)).shape == (120, 320, 3)


def test_folder_skips_bad_files(tmp_path):
    media.write_image(str(tmp_path / "a.png"), stripes())
    (tmp_path / "b.jpg").write_bytes(b"not an image")
    ok, bad = Converter(BlobDepth()).convert_folder(str(tmp_path), str(tmp_path / "out"),
                                                    StereoSettings(), OutputSettings())
    assert len(ok) == 1 and len(bad) == 1 and os.path.exists(ok[0])


def _make_video(path, frames=24, fps=12):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(frames):
        vw.write(np.roll(stripes(), i, axis=1))
    vw.release()


@needs_ffmpeg
def test_video_sample_range_and_progress(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_video(src)
    seen = []
    ok = Converter(BlobDepth()).convert_video(
        str(src), str(dst), StereoSettings(), OutputSettings(OutputFormat.SBS_HALF),
        start_frame=4, max_frames=10, prefer_nvenc=False, progress=seen.append)
    assert ok and seen[-1].done == 10
    with media.VideoReader(str(dst)) as r:
        assert r.size == (160, 120) and r.frame_count == 10
    assert not os.path.exists(str(dst) + ".video.mp4")


@needs_ffmpeg
def test_video_cancel_leaves_no_files(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_video(src)
    cancel = threading.Event()
    ok = Converter(BlobDepth()).convert_video(
        str(src), str(dst), StereoSettings(), OutputSettings(), prefer_nvenc=False,
        progress=lambda p: cancel.set() if p.done == 3 else None, cancel=cancel)
    assert not ok
    assert sorted(os.listdir(tmp_path)) == ["in.mp4"]


@pytest.mark.model
def test_real_default_model_depth_is_sane():
    from engine import load_model
    img = stripes(360, 640)
    depth = load_model("da2-small").estimate(img)
    assert depth.shape == img.shape[:2] and depth.dtype == np.float32
    assert 0.0 <= depth.min() and depth.max() <= 1.0 and depth.std() > 0.01
