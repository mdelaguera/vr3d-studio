"""
All file IO: unicode-safe images, prefetching video reader, threaded ffmpeg writer, audio mux.
The only module that talks to ffmpeg.
"""

import os
import queue
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache

import cv2
import numpy as np

from .settings import OutputFormat

_STOP = object()


# ---------- images ----------

def read_image(path: str) -> np.ndarray:
    """cv2.imread fails on non-ASCII Windows paths; decoding from bytes does not."""
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        img = None
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def write_image(path: str, img: np.ndarray, jpeg_quality: int = 95):
    ext = os.path.splitext(path)[1].lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}: {path}")
    buf.tofile(path)


# ---------- video in ----------

def _short_path(path: str) -> str:
    """Windows 8.3 path so OpenCV's FFmpeg backend can open non-ASCII paths."""
    if os.name == "nt":
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024):
            return buf.value
    return path


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self._cap = cv2.VideoCapture(_short_path(path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 and not np.isnan(fps) else 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read_at(self, index: int):
        """Single random-access frame (for scrubbing). Returns None past the end."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self, start: int = 0, limit: int = None, prefetch: int = 8):
        """Yields frames; decoding runs on a background thread so it overlaps compute."""
        if start:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        q = queue.Queue(maxsize=prefetch)
        stop = threading.Event()

        def decode():
            n = 0
            while not stop.is_set() and (limit is None or n < limit):
                ok, frame = self._cap.read()
                if not ok:
                    break
                q.put(frame)
                n += 1
            q.put(_STOP)

        t = threading.Thread(target=decode, daemon=True)
        t.start()
        try:
            while (item := q.get()) is not _STOP:
                yield item
        finally:
            stop.set()
            while t.is_alive():  # unblock the decoder if it's waiting on a full queue
                try:
                    q.get_nowait()
                except queue.Empty:
                    t.join(0.05)

    def close(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------- video out ----------

def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("FFmpeg was not found. Install it and add it to PATH (e.g. `winget install Gyan.FFmpeg`).")
    return exe


# GPU HEVC encoders by vendor, with roughly matched quality settings. First one that works wins.
HW_ENCODERS = [
    ("hevc_nvenc", ["-preset", "p4", "-cq", "22"]),                        # NVIDIA
    ("hevc_amf", ["-quality", "quality", "-rc", "cqp", "-qp_i", "22", "-qp_p", "22"]),  # AMD
    ("hevc_qsv", ["-global_quality", "22"]),                               # Intel
]
CPU_ENCODER = ["-c:v", "libx264", "-crf", "19", "-preset", "fast"]


def _encoder_works(name: str) -> bool:
    """Listed != usable (needs the right GPU + driver), so probe with a tiny real encode."""
    try:
        r = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", name, "-f", "null", "-"],
            capture_output=True, timeout=20,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def hardware_encoder() -> str:
    """Name of the first working GPU encoder on this machine, or '' if none."""
    return next((name for name, _ in HW_ENCODERS if _encoder_works(name)), "")


def encoder_args(prefer_hardware: bool = True) -> list:
    name = hardware_encoder() if prefer_hardware else ""
    if name:
        return ["-c:v", name, *dict(HW_ENCODERS)[name], "-tag:v", "hvc1"]
    return CPU_ENCODER


class VideoWriter:
    """Pipes raw BGR frames into ffmpeg from a background thread (encoding overlaps compute)."""

    def __init__(self, path: str, size, fps: float, prefer_hardware: bool = True, queue_size: int = 8):
        w, h = size
        self.path = path
        self._log = tempfile.TemporaryFile()
        cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               *encoder_args(prefer_hardware), "-pix_fmt", "yuv420p", path]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)
        self._q = queue.Queue(maxsize=queue_size)
        self._error = None
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while (frame := self._q.get()) is not _STOP:
            if self._error is None:
                try:
                    self._proc.stdin.write(frame.tobytes())
                except OSError as exc:  # ffmpeg died; keep draining so write() never blocks
                    self._error = exc

    def write(self, frame: np.ndarray):
        if self._error is not None:
            raise RuntimeError(f"Video encoder failed: {self._ffmpeg_log()}")
        self._q.put(frame)

    def _ffmpeg_log(self) -> str:
        self._log.seek(0)
        return self._log.read().decode("utf-8", "replace").strip()[-800:] or str(self._error)

    def close(self):
        self._q.put(_STOP)
        self._thread.join()
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        code = self._proc.wait()
        log = self._ffmpeg_log() if code != 0 or self._error else ""
        self._log.close()
        if code != 0 or self._error:
            raise RuntimeError(f"Video encoding failed: {log}")


_STEREO_METADATA = {
    OutputFormat.VR180: ["-metadata:s:v:0", "stereo_mode=left_right",
                         "-metadata:s:v:0", "projection=equirectangular",
                         "-metadata:s:v:0", "spherical-video=true"],
    OutputFormat.SBS_FULL: ["-metadata:s:v:0", "stereo_mode=left_right"],
    OutputFormat.SBS_HALF: ["-metadata:s:v:0", "stereo_mode=left_right"],
}


def mux_audio(source: str, video_only: str, dest: str, fmt: OutputFormat,
              start_sec: float = 0.0, duration_sec: float = None):
    """Copy the source's audio (trimmed to the rendered range) next to the new video stream."""
    trim = (["-ss", f"{start_sec:.3f}"] if start_sec > 0 else []) + (["-t", f"{duration_sec:.3f}"] if duration_sec else [])
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", video_only, *trim, "-i", source,
           "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
           *_STEREO_METADATA.get(fmt, []), "-movflags", "+faststart", dest]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace").strip()[-800:])
