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
DIRECTML_CMD = "pip uninstall -y onnxruntime; pip install onnxruntime-directml"
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
    backend: str = ""                # engine device: cuda | mps | dml | cpu
    has_gpu: bool = False            # the default model's AI runs on a GPU
    encoder: str = ""                # human label for video encoding, e.g. "AMD GPU"
    issues: list = field(default_factory=list)

    def runs_on_gpu(self, model_info) -> bool:
        """DirectML only accelerates models that ship an ONNX export; CUDA/Metal accelerate all."""
        return self.backend in ("cuda", "mps") or (self.backend == "dml" and bool(model_info.onnx))


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
            f"Your {other} isn't being used for AI",
            "Install ONNX Runtime with DirectML to run Depth Anything V2 Small on your GPU (about 30× faster), "
            "then restart Any2VR.", DIRECTML_CMD)
    return Issue(
        "Running on CPU",
        "No supported GPU was found. Photos convert fine; long videos will be slow. "
        "Depth Anything V2 Small is the fastest model for this computer.")


def check() -> SystemReport:
    from engine.depth import device_label, pick_device  # imports torch: keep off the UI thread
    from engine.media import hardware_encoder

    report = SystemReport(device_label=device_label(), backend=pick_device())
    report.has_gpu = report.backend != "cpu"
    if report.backend in ("dml", "cpu"):
        adapters = _display_adapters()
        if report.backend == "dml":
            report.device_label = f"{adapters[0] if adapters else 'GPU'} · DirectML"
            nvidia = next((a for a in adapters if "nvidia" in a.lower()), "")
            if nvidia:  # works via DirectML, but CUDA is faster and accelerates every model
                report.issues.append(Issue(
                    f"Get more speed from your {nvidia}",
                    "The fast model already uses your GPU via DirectML. Installing PyTorch with CUDA makes it faster "
                    "and also accelerates the slow, high-quality models.", CUDA_TORCH_CMD))
        else:
            report.issues.append(_gpu_issue(adapters))

    if shutil.which("ffmpeg") is None:
        report.issues.append(Issue(
            "FFmpeg is missing",
            "Video conversion needs FFmpeg. Photos still work. Install it, then restart Any2VR.",
            FFMPEG_CMD, blocking=True))
    else:
        enc = hardware_encoder()
        report.encoder = f"{ENCODER_VENDORS[enc]} GPU" if enc else "CPU"
    return report
