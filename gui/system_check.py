"""
Setup checks with plain-English fixes. Run once at startup on a background thread.
No Tk imports here so it stays testable.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass, field

CUDA_TORCH_CMD = "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
FFMPEG_CMD = "winget install Gyan.FFmpeg"
VIRTUAL_ADAPTERS = ("virtual", "basic display", "basic render", "remote display", "parsec", "citrix")
ENCODER_VENDORS = {"hevc_nvenc": "NVIDIA", "hevc_amf": "AMD", "hevc_qsv": "Intel"}


@dataclass
class Issue:
    title: str
    detail: str
    fix_command: str = ""
    blocking: bool = False   # True = some features won't work at all


@dataclass
class SystemReport:
    device_label: str = "Checking hardware..."
    has_gpu: bool = False            # the AI runs on a GPU
    encoder: str = ""                # human label for video encoding, e.g. "AMD GPU"
    issues: list = field(default_factory=list)


def _display_adapters() -> list:
    """Physical graphics adapters Windows knows about (driver-independent)."""
    if sys.platform != "win32":
        return []
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    names = [n.strip() for n in r.stdout.splitlines() if n.strip()]
    return [n for n in names if not any(v in n.lower() for v in VIRTUAL_ADAPTERS)]


def _gpu_issue(adapters: list):
    """Why the AI isn't on a GPU, and what (if anything) the user can do about it."""
    nvidia = next((a for a in adapters if "nvidia" in a.lower()), "")
    if nvidia:
        return Issue(
            f"Your {nvidia} isn't being used",
            "PyTorch was installed without CUDA support, so the AI runs on your CPU (about 10-50× slower). "
            "Reinstall PyTorch with CUDA, then restart Any2VR.",
            CUDA_TORCH_CMD)
    other = next((a for a in adapters if any(v in a.lower() for v in ("amd", "radeon", "intel", "arc"))), "")
    if other:
        return Issue(
            f"The AI runs on your CPU ({other} not supported yet)",
            f"Any2VR's AI accelerates on NVIDIA and Apple GPUs today. Your {other} still speeds up video encoding. "
            "Photos convert fine; long videos will be slow. Depth Anything V2 Small is the fastest model here.")
    return Issue(
        "Running on CPU",
        "No supported GPU was found. Photos convert fine; long videos will be slow. "
        "Depth Anything V2 Small is the fastest model for this computer.")


def check() -> SystemReport:
    from engine.depth import device_label, pick_device  # imports torch: keep off the UI thread
    from engine.media import hardware_encoder

    report = SystemReport(device_label=device_label())
    report.has_gpu = pick_device() != "cpu"
    if not report.has_gpu:
        report.issues.append(_gpu_issue(_display_adapters()))

    if shutil.which("ffmpeg") is None:
        report.issues.append(Issue(
            "FFmpeg is missing",
            "Video conversion needs FFmpeg. Photos still work. Install it, then restart Any2VR.",
            FFMPEG_CMD, blocking=True))
    else:
        enc = hardware_encoder()
        report.encoder = f"{ENCODER_VENDORS[enc]} GPU" if enc else "CPU"
    return report
