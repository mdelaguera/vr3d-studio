"""Command line: convert a photo, video, or folder of photos."""

import argparse
import os
import sys
import time

from . import (CATALOG, DEFAULT_MODEL, Converter, OutputFormat, OutputSettings, StereoSettings,
               device_label, is_video, load_model, output_path_for)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Convert 2D photos and videos to 3D (SBS, VR180, anaglyph).")
    p.add_argument("input", nargs="?", help="Photo, video, or folder of photos (omit to open the app)")
    p.add_argument("-i", "--input", dest="input_flag", help=argparse.SUPPRESS)  # old flag, still accepted
    p.add_argument("-o", "--output", help="Output file (or folder for albums). Default: next to the input")
    p.add_argument("-m", "--mode", default="sbs_full", choices=[f.value for f in OutputFormat])
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(CATALOG), help="Depth model (see --list-models)")
    p.add_argument("--strength", "--ipd", type=float, default=0.035, help="3D strength (default 0.035 = 'Natural')")
    p.add_argument("--focus", "--conv", type=float, default=None,
                   help="Fixed focus plane 0.1-0.9 (default: automatic focus on the subject)")
    p.add_argument("--fov", type=float, default=110.0, help="VR180 field of view in degrees")
    p.add_argument("--swap-eyes", action="store_true")
    p.add_argument("--no-nvenc", action="store_true", help="Encode on the CPU (libx264) instead of NVIDIA NVENC")
    p.add_argument("--no-smoothing", "--no-temporal", action="store_true", help="Disable video depth smoothing")
    p.add_argument("--list-models", action="store_true", help="Show available depth models and their licenses")
    return p


def _progress(p):
    eta = time.strftime("%H:%M:%S", time.gmtime(p.eta_seconds))
    print(f"\r  {p.done}/{p.total} ({p.fraction:.0%}) · {p.rate:.1f}/s · ETA {eta}   ", end="", flush=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_models:
        for m in CATALOG.values():
            print(f"{m.id:14s} {m.name:38s} {m.license:11s} {m.speed:7s} ~{m.download_mb} MB  {m.blurb}")
        return 0

    src = args.input or args.input_flag
    if not src:
        build_parser().print_usage()
        return 2
    if not os.path.exists(src):
        print(f"Input not found: {src}", file=sys.stderr)
        return 1

    fmt = OutputFormat(args.mode)
    stereo = StereoSettings(strength=args.strength, focus=0.5 if args.focus is None else args.focus,
                            auto_focus=args.focus is None, swap_eyes=args.swap_eyes)
    output = OutputSettings(fmt, vr_fov=args.fov)

    print(f"Loading {CATALOG[args.model].name} on {device_label()}...")
    conv = Converter(load_model(args.model))

    if os.path.isdir(src):
        dst = args.output or os.path.join(src, "3D")
        ok, bad = conv.convert_folder(src, dst, stereo, output, progress=_progress)
        print(f"\nConverted {len(ok)} photos into {dst}" + (f" ({len(bad)} skipped)" if bad else ""))
    elif is_video(src):
        dst = args.output or output_path_for(src, fmt, ext=".mp4")
        conv.convert_video(src, dst, stereo, output, smooth=not args.no_smoothing,
                           prefer_nvenc=not args.no_nvenc, progress=_progress)
        print(f"\nSaved {dst}")
    else:
        dst = args.output or output_path_for(src, fmt)
        conv.convert_image(src, dst, stereo, output)
        print(f"Saved {dst}")
    return 0
