"""User preferences, option tables, and the recent-outputs list (one JSON file in the home folder)."""

import json
import os
import time

from engine import CATALOG, DEFAULT_MODEL

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".any2vr.json")
LEGACY_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".vr3d_studio.json")  # pre-rename
MAX_RECENT = 8

# Dropdown label -> model id, built from the engine catalog so new models appear automatically
MODELS = {f"{m.name}  ·  {m.speed}  ·  {m.license}": m.id for m in CATALOG.values()}

MODES = {
    "VR180 3D (Quest / Pico headsets)": "vr180",
    "Full SBS 3D (3D TV / monitor)": "sbs_full",
    "Half SBS 3D (half width)": "sbs_half",
    "Anaglyph (red / cyan glasses)": "anaglyph",
    "Depth map only": "depth_only",
}

# name -> (strength, focus)
PROFILES = {
    "Soft - relaxed (2.0%)": (0.020, 0.50),
    "Natural - comfortable (3.5%)": (0.035, 0.50),
    "Deep - dynamic (5.0%)": (0.050, 0.35),
    "Pop-out - leaps off screen (6.5%)": (0.065, 0.15),
    "Extreme (8.0%)": (0.080, 0.10),
}

DEFAULTS = {
    "model": DEFAULT_MODEL,
    "mode": "vr180",
    "profile": "Natural - comfortable (3.5%)",
    "ipd": 0.035,
    "conv": 0.50,
    "fov": 110.0,
    "auto_conv": True,
    "swap": False,
    "temporal": True,
    "nvenc": True,
    "appearance": "Dark",
    "welcomed": False,
    "recent": [],       # [{"path": str, "kind": "video"|"image"|"folder", "time": epoch}]
}


def label_for(mapping: dict, value) -> str:
    return next(k for k, v in mapping.items() if v == value)


def load() -> dict:
    settings = dict(DEFAULTS)
    path = SETTINGS_PATH if os.path.exists(SETTINGS_PATH) else LEGACY_SETTINGS_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
    except (OSError, ValueError, AttributeError):
        pass
    # Drop stale values that no longer match an option (e.g. removed models)
    for key, options in (("model", MODELS.values()), ("mode", MODES.values()), ("profile", PROFILES),
                         ("appearance", ("Dark", "Light", "System"))):
        if settings[key] not in options:
            settings[key] = DEFAULTS[key]
    settings["recent"] = prune_recent(settings["recent"])
    return settings


def save(settings: dict):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump({k: settings[k] for k in DEFAULTS if k in settings}, fh, indent=2)
    except OSError as exc:
        print(f"[Settings save error] {exc}")


def prune_recent(recent) -> list:
    """Keep well-formed entries whose files still exist."""
    if not isinstance(recent, list):
        return []
    return [r for r in recent if isinstance(r, dict) and os.path.exists(str(r.get("path", "")))][:MAX_RECENT]


def add_recent(recent: list, path: str, kind: str) -> list:
    entry = {"path": path, "kind": kind, "time": int(time.time())}
    return [entry] + [r for r in recent if r.get("path") != path][: MAX_RECENT - 1]
