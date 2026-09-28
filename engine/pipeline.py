"""
Converter: the single entry point GUI, CLI and cloud workers use.
Frame -> image file -> video file -> folder, with progress callbacks and cancellation.
"""

import os
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import media
from .depth import DepthModel
from .settings import OutputFormat, OutputSettings, StereoSettings, is_image, output_path_for
from .stereo import StereoRenderer, anaglyph, depth_colormap, side_by_side
from .temporal import DepthSmoother
from .vr180 import vr180_sbs


@dataclass
class Progress:
    done: int
    total: int
    started: float

    @property
    def fraction(self) -> float:
        return self.done / max(1, self.total)

    @property
    def rate(self) -> float:
        return self.done / max(1e-6, time.time() - self.started)

    @property
    def eta_seconds(self) -> float:
        return (self.total - self.done) / self.rate if self.rate > 0 else 0.0


ProgressFn = Optional[Callable[[Progress], None]]


class Cancelled(Exception):
    pass


def compose(fmt: OutputFormat, left, right, depth, output: OutputSettings) -> np.ndarray:
    """Pack a stereo pair (or depth) into the requested output layout."""
    if fmt is OutputFormat.DEPTH:
        return depth_colormap(depth)
    if fmt is OutputFormat.SBS_FULL:
        return side_by_side(left, right)
    if fmt is OutputFormat.SBS_HALF:
        return side_by_side(left, right, half=True)
    if fmt is OutputFormat.ANAGLYPH:
        return anaglyph(left, right)
    return vr180_sbs(left, right, output.vr_eye_size, output.vr_fov)


class Converter:
    def __init__(self, depth_model: DepthModel):
        self.depth_model = depth_model
        self.renderer = StereoRenderer()

    # ----- single frame -----

    def stereo_pair(self, bgr, depth, stereo: StereoSettings):
        return self.renderer.render(bgr, depth, stereo)

    def render_frame(self, bgr, stereo: StereoSettings, output: OutputSettings, depth=None) -> np.ndarray:
        depth = self.depth_model.estimate(bgr) if depth is None else depth
        if output.format is OutputFormat.DEPTH:
            return depth_colormap(depth)
        left, right = self.renderer.render(bgr, depth, stereo)
        return compose(output.format, left, right, depth, output)

    # ----- files -----

    def convert_image(self, src: str, dst: str, stereo: StereoSettings, output: OutputSettings):
        media.write_image(dst, self.render_frame(media.read_image(src), stereo, output))

    def convert_video(self, src: str, dst: str, stereo: StereoSettings, output: OutputSettings, *,
                      smooth: bool = True, prefer_nvenc: bool = True, start_frame: int = 0,
                      max_frames: int = None, progress: ProgressFn = None,
                      cancel: threading.Event = None) -> bool:
        """Returns False if cancelled. The output keeps the source audio for the rendered range."""
        smoother = DepthSmoother() if smooth else None
        temp = dst + ".video.mp4"
        writer = None
        done = 0
        try:
            with media.VideoReader(src) as reader:
                total = max(0, reader.frame_count - start_frame)
                total = min(total, max_frames) if max_frames else total
                started = time.time()
                # closing() stops the decode thread before the reader is released, even on errors
                with closing(reader.frames(start_frame, max_frames)) as frames:
                    for frame in frames:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled
                        depth = self.depth_model.estimate(frame)
                        if smoother:
                            depth = smoother(frame, depth)
                        out = self.render_frame(frame, stereo, output, depth=depth)
                        if writer is None:  # size known after the first frame
                            writer = media.VideoWriter(temp, (out.shape[1], out.shape[0]), reader.fps, prefer_nvenc)
                        writer.write(out)
                        done += 1
                        if progress:
                            progress(Progress(done, max(total, done), started))
                fps = reader.fps
            if writer is None:
                raise ValueError(f"No frames could be read from: {src}")
            writer.close()
            writer = None
            try:
                media.mux_audio(src, temp, dst, output.format, start_frame / fps,
                                done / fps if max_frames else None)
            except RuntimeError as exc:  # never lose a finished render over audio
                print(f"[Audio mux failed, saving video without audio] {exc}")
                os.replace(temp, dst)
            return True
        except Cancelled:
            return False
        finally:
            if writer is not None:
                try:
                    writer.close()
                except RuntimeError:
                    pass
            if os.path.exists(temp):
                os.remove(temp)

    def convert_folder(self, src_dir: str, dst_dir: str, stereo: StereoSettings, output: OutputSettings, *,
                       progress: ProgressFn = None, cancel: threading.Event = None):
        """Converts every image in a folder. Bad files are skipped. Returns (converted, failed) paths."""
        files = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir) if is_image(f))
        os.makedirs(dst_dir, exist_ok=True)
        converted, failed = [], []
        started = time.time()
        for i, src in enumerate(files):
            if cancel is not None and cancel.is_set():
                break
            dst = output_path_for(src, output.format, dst_dir)
            try:
                self.convert_image(src, dst, stereo, output)
                converted.append(dst)
            except Exception as exc:
                print(f"[Skipped {os.path.basename(src)}] {exc}")
                failed.append(src)
            if progress:
                progress(Progress(i + 1, len(files), started))
        return converted, failed
