"""
VR3D Studio - Graphical User Interface
Built with CustomTkinter. Multi-threaded live preview, video timeline scrubber,
switchable 3D preview views, and persistent settings.
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.io_utils import imread_safe
from core.depth_estimator import DepthEstimator
from core.stereo_warper import StereoWarper
from core.vr180_projector import VR180Projector
from core.video_processor import VideoProcessor

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".vr3d_studio.json")

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# VR headset naming suffixes (Pico, Quest, Skybox VR auto-detection)
SUFFIX_MAP = {
    "vr180": "_180_SBS",
    "sbs_full": "_3DH_SBS",
    "sbs_half": "_3DH_Half_SBS",
    "anaglyph": "_3D_Anaglyph",
    "depth_only": "_Depth",
}

MODELS = {
    "Hybrid fusion (Depth Anything + Marigold)": "hybrid",
    "Depth Anything V2 Small (fastest)": "vits",
    "Depth Anything V2 Base (balanced)": "vitb",
    "Depth Anything V2 Large (best quality)": "vitl",
    "Marigold LCM (diffusion depth)": "marigold",
}

MODES = {
    "VR180 3D (Quest / Pico headsets)": "vr180",
    "Full SBS 3D (3D TV / monitor)": "sbs_full",
    "Half SBS 3D (half width)": "sbs_half",
    "Anaglyph (red / cyan glasses)": "anaglyph",
    "Depth map only": "depth_only",
}

# name -> (strength / ipd, focus / convergence)
PROFILES = {
    "Soft - relaxed (2.0%)": (0.020, 0.50),
    "Natural - comfortable (3.5%)": (0.035, 0.50),
    "Deep - dynamic (5.0%)": (0.050, 0.35),
    "Pop-out - leaps off screen (6.5%)": (0.065, 0.15),
    "Extreme (8.0%)": (0.080, 0.10),
}

# (internal key, button label)
VIEWS = [
    ("wiggle", "Wiggle 3D"),
    ("sbs", "SBS"),
    ("vr180", "VR180"),
    ("anaglyph", "Anaglyph"),
    ("depth", "Depth"),
    ("left", "Left eye"),
    ("right", "Right eye"),
    ("original", "Original"),
]
VIEW_LABELS = {k: v for k, v in VIEWS}
VIEW_KEYS = {v: k for k, v in VIEWS}

# Palette: (light, dark) tuples follow the appearance mode automatically
ACCENT = ("#5B5BD6", "#7C7CF0")
ACCENT_HOVER = ("#4A4AC4", "#6868DE")
GO = ("#1F9D6B", "#23B47A")
GO_HOVER = ("#1A8A5D", "#1E9C6A")
DANGER = ("#D64545", "#E05A5A")
DANGER_HOVER = ("#BF3B3B", "#C94C4C")
SUBTLE = ("#E4E6EE", "#2A2D3A")
SUBTLE_HOVER = ("#D5D8E3", "#343849")
CARD = ("#F4F5F9", "#1D1F29")
SIDEBAR = ("#FFFFFF", "#15161E")
MAIN_BG = ("#EBEDF3", "#0F1016")
MUTED = ("#6B7080", "#8B90A3")
TEXT = ("#1B1D26", "#E8E9F0")
PREVIEW_BG = "#0B0C10"

DEFAULTS = {
    "model": "vits",
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
}


def load_settings() -> dict:
    settings = dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    # Drop stale values that no longer match an option
    if settings["model"] not in MODELS.values():
        settings["model"] = DEFAULTS["model"]
    if settings["mode"] not in MODES.values():
        settings["mode"] = DEFAULTS["mode"]
    if settings["profile"] not in PROFILES:
        settings["profile"] = DEFAULTS["profile"]
    if settings["appearance"] not in ("Dark", "Light", "System"):
        settings["appearance"] = DEFAULTS["appearance"]
    return settings


def label_for(mapping: dict, value) -> str:
    return next(k for k, v in mapping.items() if v == value)


class VR3DStudioApp(ctk.CTk):
    def __init__(self):
        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings["appearance"])
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title("VR3D Studio - 2D to 3D SBS & VR180 Converter")
        self.geometry("1280x800")
        self.minsize(980, 640)
        self.configure(fg_color=MAIN_BG)

        # Core engines
        self.depth_estimator = None
        self.stereo_warper = StereoWarper()
        self.video_processor = None

        # State
        self.input_file_path = ""
        self.output_file_path = ""
        self.is_video = False
        self.is_folder = False
        self.album_files = []
        self.total_video_frames = 0
        self.video_fps = 30.0
        self.current_frame_bgr = None
        self.cached_depth = None
        self.cached_left = None
        self.cached_right = None
        self.active_view = "wiggle"
        self.is_processing = False
        self.wiggle_eye = 0
        self.wiggle_job = None
        self.preview_job = None
        self.scrub_job = None
        self.preview_busy = False
        self.preview_again = False

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._init_backend_async()

    # ==========================================
    # Backend
    # ==========================================
    def _init_backend_async(self):
        """Loads the AI model in a background thread so the UI starts immediately."""
        self._set_status("Loading depth model onto GPU...")
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            model_key = MODELS.get(self.model_var.get(), "vits")
            self.depth_estimator = DepthEstimator(model_size=model_key)
            self.video_processor = VideoProcessor(self.depth_estimator, self.stereo_warper)
            gpu = self._detect_gpu()
            self.after(0, lambda: self.hw_label.configure(text=gpu))
            self.after(0, lambda: self._set_status("Ready."))
            self.after(0, self._on_settings_change)
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Model load error] {err_msg}")
            self.after(0, lambda msg=err_msg: self._set_status(f"Model error: {msg}"))

    @staticmethod
    def _detect_gpu() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return f"⚡ {torch.cuda.get_device_name(0)} · CUDA"
        except Exception:
            pass
        return "CPU mode (no CUDA GPU found)"

    # ==========================================
    # UI construction
    # ==========================================
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    def _card(self, parent, title):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12)
        card.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(
            card, text=title.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color=MUTED
        ).pack(padx=12, pady=(10, 4), anchor="w")
        return card

    def _field_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=12), text_color=TEXT).pack(padx=12, anchor="w")

    def _option_menu(self, parent, values, variable, command):
        menu = ctk.CTkOptionMenu(
            parent, values=values, variable=variable, command=command, height=30,
            fg_color=SUBTLE, button_color=SUBTLE_HOVER, button_hover_color=ACCENT_HOVER,
            text_color=TEXT, dynamic_resizing=False, corner_radius=8,
        )
        menu.pack(fill="x", padx=12, pady=(2, 8))
        return menu

    def _slider_row(self, parent, title, from_, to, steps, value, command):
        """Title on the left, live value on the right, slider underneath."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(2, 8))
        head = ctk.CTkFrame(row, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text=title, font=ctk.CTkFont(size=12), text_color=TEXT).pack(side="left")
        value_lbl = ctk.CTkLabel(head, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color=ACCENT)
        value_lbl.pack(side="right")
        slider = ctk.CTkSlider(
            row, from_=from_, to=to, number_of_steps=steps, command=command,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER, progress_color=ACCENT,
        )
        slider.set(value)
        slider.pack(fill="x", pady=(2, 0))
        return row, value_lbl, slider

    def _switch(self, parent, text, on, command=None):
        sw = ctk.CTkSwitch(parent, text=text, font=ctk.CTkFont(size=12), text_color=TEXT,
                           progress_color=ACCENT, command=command)
        if on:
            sw.select()
        sw.pack(padx=12, pady=3, anchor="w")
        return sw

    def _build_sidebar(self):
        s = self.settings
        sidebar = ctk.CTkFrame(self, width=330, corner_radius=0, fg_color=SIDEBAR)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(1, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        # --- Header ---
        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        ctk.CTkLabel(header, text="VR3D Studio", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(header, text="Turn 2D photos & videos into 3D / VR180",
                     font=ctk.CTkFont(size=12), text_color=MUTED).pack(anchor="w")
        self.hw_label = ctk.CTkLabel(
            header, text="Detecting GPU...", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=GO, fg_color=CARD, corner_radius=10, height=24,
        )
        self.hw_label.pack(anchor="w", pady=(8, 0), ipadx=8)

        # --- Scrollable settings ---
        body = ctk.CTkScrollableFrame(sidebar, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=4)

        # Source
        src = self._card(body, "1  Source")
        btn_row = ctk.CTkFrame(src, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(2, 4))
        btn_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            btn_row, text="Open file", height=32, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._browse_input,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(
            btn_row, text="Open folder", height=32, corner_radius=8,
            fg_color=SUBTLE, hover_color=SUBTLE_HOVER, text_color=TEXT, command=self._browse_folder,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.lbl_input_path = ctk.CTkLabel(
            src, text="No file selected  ·  Ctrl+O", text_color=MUTED, wraplength=270,
            justify="left", font=ctk.CTkFont(size=11),
        )
        self.lbl_input_path.pack(padx=12, pady=(0, 10), anchor="w")

        # Model & format
        fmt = self._card(body, "2  Model & format")
        self._field_label(fmt, "Depth model")
        self.model_var = ctk.StringVar(value=label_for(MODELS, s["model"]))
        self._option_menu(fmt, list(MODELS), self.model_var, self._on_model_changed)
        self._field_label(fmt, "Output format")
        self.mode_var = ctk.StringVar(value=label_for(MODES, s["mode"]))
        self._option_menu(fmt, list(MODES), self.mode_var, self._on_mode_changed)

        # 3D effect
        fx = self._card(body, "3  3D effect")
        self._field_label(fx, "Preset")
        self.profile_var = ctk.StringVar(value=s["profile"])
        self._option_menu(fx, list(PROFILES), self.profile_var, self._on_profile_changed)
        _, self.lbl_ipd, self.slider_ipd = self._slider_row(
            fx, "Depth strength", 0.010, 0.090, 80, s["ipd"], self._on_ipd_slide)
        _, self.lbl_conv, self.slider_conv = self._slider_row(
            fx, "Focus plane", 0.1, 0.9, 80, s["conv"], self._on_conv_slide)
        self.chk_auto_conv = self._switch(fx, "Auto focus on subject", s["auto_conv"], self._on_stereo_toggle)
        self.fov_row, self.lbl_fov, self.slider_fov = self._slider_row(
            fx, "VR field of view", 80.0, 150.0, 70, s["fov"], self._on_fov_slide)
        ctk.CTkFrame(fx, height=4, fg_color="transparent").pack()
        self._fov_anchor = self.chk_auto_conv

        # Options
        opts = self._card(body, "4  Options")
        self.chk_swap_eyes = self._switch(opts, "Swap left / right eyes", s["swap"], self._on_stereo_toggle)
        self.chk_temporal = self._switch(opts, "Anti-flicker smoothing (video)", s["temporal"])
        self.chk_nvenc = self._switch(opts, "NVIDIA NVENC hardware encoding", s["nvenc"])
        ctk.CTkFrame(opts, height=6, fg_color="transparent").pack()

        # Appearance
        look = self._card(body, "Appearance")
        self.appearance_seg = ctk.CTkSegmentedButton(
            look, values=["Dark", "Light", "System"], command=self._on_appearance_changed,
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
        )
        self.appearance_seg.set(s["appearance"])
        self.appearance_seg.pack(fill="x", padx=12, pady=(2, 12))

        # Initialise value labels
        self._on_ipd_slide(s["ipd"], refresh=False)
        self._on_conv_slide(s["conv"], refresh=False)
        self._on_fov_slide(s["fov"], refresh=False)
        self._update_fov_visibility()

        # --- Sticky action bar (always visible) ---
        actions = ctk.CTkFrame(sidebar, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 16))
        actions.grid_columnconfigure((0, 1), weight=1)

        self.btn_preview = ctk.CTkButton(
            actions, text="Refresh preview", height=32, corner_radius=8,
            fg_color=SUBTLE, hover_color=SUBTLE_HOVER, text_color=TEXT,
            command=self._update_preview_manual,
        )
        self.btn_preview.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
        self.btn_quick_test = ctk.CTkButton(
            actions, text="5 s sample", height=32, corner_radius=8,
            fg_color=SUBTLE, hover_color=SUBTLE_HOVER, text_color=TEXT,
            command=self._start_quick_test,
        )
        self.btn_quick_test.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))

        self.btn_start = ctk.CTkButton(
            actions, text="Convert  (Ctrl+Enter)", height=42, corner_radius=10,
            fg_color=GO, hover_color=GO_HOVER, font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_conversion,
        )
        self.btn_start.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.btn_cancel = ctk.CTkButton(
            actions, text="Cancel  (Esc)", height=42, corner_radius=10,
            fg_color=DANGER, hover_color=DANGER_HOVER, font=ctk.CTkFont(size=14, weight="bold"),
            command=self._cancel_conversion,
        )
        # Cancel replaces Convert while busy (same grid cell)

    def _build_main(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # View switcher
        self.view_seg = ctk.CTkSegmentedButton(
            main, values=[label for _, label in VIEWS], command=self._on_view_selected,
            height=32, selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.view_seg.set(VIEW_LABELS[self.active_view])
        self.view_seg.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Preview canvas
        self.canvas_frame = ctk.CTkFrame(main, fg_color=PREVIEW_BG, corner_radius=12)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.lbl_image = tk.Label(
            self.canvas_frame, bg=PREVIEW_BG, fg="#8B90A3", font=("Segoe UI", 13),
            text="Open a video, photo, or photo folder to get started\n\n"
                 "Ctrl+O  open file     ·     Ctrl+Shift+O  open folder\n"
                 "1-8  switch view     ·     F5  refresh preview",
        )
        self.lbl_image.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.canvas_frame.bind("<Configure>", lambda evt: self._render_current_view())

        # Video timeline scrubber (videos only)
        self.scrub_frame = ctk.CTkFrame(main, fg_color="transparent")
        self.scrub_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.scrub_frame.grid_columnconfigure(1, weight=1)
        self.lbl_scrub_time = ctk.CTkLabel(self.scrub_frame, text="00:00 / 00:00", width=96,
                                           font=ctk.CTkFont(size=12), text_color=MUTED)
        self.lbl_scrub_time.grid(row=0, column=0, padx=(0, 10))
        self.slider_scrub = ctk.CTkSlider(
            self.scrub_frame, from_=0, to=100, command=self._on_scrub,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER, progress_color=ACCENT,
        )
        self.slider_scrub.set(0)
        self.slider_scrub.grid(row=0, column=1, sticky="ew")
        self.scrub_frame.grid_remove()

        # Status bar
        bar = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12)
        bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(bar, text="Starting...", font=ctk.CTkFont(size=12), text_color=TEXT)
        self.status_label.grid(row=0, column=0, sticky="w", padx=14, pady=(8, 2))
        self.stats_label = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color=ACCENT)
        self.stats_label.grid(row=0, column=1, sticky="e", padx=14, pady=(8, 2))
        self.progress_bar = ctk.CTkProgressBar(bar, height=6, progress_color=ACCENT, fg_color=SUBTLE)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 10))

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self._browse_input())
        self.bind("<Control-O>", lambda e: self._browse_folder())  # Ctrl+Shift+O
        self.bind("<Control-Return>", lambda e: None if self.is_processing else self._start_conversion())
        self.bind("<Escape>", lambda e: self._cancel_conversion() if self.is_processing else None)
        self.bind("<F5>", lambda e: self._update_preview_manual())
        for i, (key, label) in enumerate(VIEWS, start=1):
            self.bind(str(i), lambda e, k=key: self._select_view(k))

    # ==========================================
    # Small helpers
    # ==========================================
    def _set_status(self, text):
        self.status_label.configure(text=text)

    def _mode_key(self):
        return MODES.get(self.mode_var.get(), "vr180")

    def _set_busy(self, busy):
        self.is_processing = busy
        state = "disabled" if busy else "normal"
        self.btn_quick_test.configure(state=state)
        self.btn_preview.configure(state=state)
        if busy:
            self.btn_start.grid_remove()
            self.btn_cancel.grid(row=1, column=0, columnspan=2, sticky="ew")
            self.progress_bar.set(0.0)
        else:
            self.btn_cancel.grid_remove()
            self.btn_start.grid()

    def _gather_params(self):
        return dict(
            mode=self._mode_key(),
            ipd_offset=self.slider_ipd.get(),
            convergence=self.slider_conv.get(),
            h_fov=self.slider_fov.get(),
            swap_eyes=self.chk_swap_eyes.get() == 1,
            auto_convergence=self.chk_auto_conv.get() == 1,
        )

    def _offer_open_folder(self, title, message, folder):
        if messagebox.askyesno(title, f"{message}\n\nOpen the output folder?"):
            try:
                os.startfile(folder)
            except Exception:
                pass

    def _switch_mode_for_photos(self):
        """VR180 is meant for video; photos look best as Full SBS."""
        if self._mode_key() == "vr180":
            sbs_label = label_for(MODES, "sbs_full")
            self.mode_var.set(sbs_label)
            self._on_mode_changed(sbs_label)

    # ==========================================
    # Event handlers
    # ==========================================
    def _browse_input(self):
        if self.is_processing:
            return
        file_path = filedialog.askopenfilename(
            title="Choose a video or photo",
            filetypes=[
                ("Media files", "*.mp4 *.mkv *.mov *.avi *.jpg *.jpeg *.png *.webp *.bmp"),
                ("Videos", "*.mp4 *.mkv *.mov *.avi"),
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.input_file_path = file_path
        self.is_folder = False
        self.album_files = []
        self.is_video = os.path.splitext(file_path)[1].lower() in VIDEO_EXTS

        if self.is_video:
            cap = cv2.VideoCapture(file_path)
            self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            ret, frame = cap.read()
            cap.release()
            if not ret:
                messagebox.showerror("Can't open video", f"Could not read frames from:\n{file_path}")
                return
            self.current_frame_bgr = frame
            self.slider_scrub.configure(to=max(1, self.total_video_frames - 1))
            self.slider_scrub.set(0)
            self._update_scrub_label(0)
            self.scrub_frame.grid()
            duration = time.strftime("%M:%S", time.gmtime(int(self.total_video_frames / self.video_fps)))
            detail = f"Video · {duration} · {self.video_fps:.0f} fps"
        else:
            self.current_frame_bgr = imread_safe(file_path)
            if self.current_frame_bgr is None:
                messagebox.showerror("Can't open image", f"Could not read image:\n{file_path}")
                return
            self.scrub_frame.grid_remove()
            self._switch_mode_for_photos()
            h, w = self.current_frame_bgr.shape[:2]
            detail = f"Photo · {w}×{h}"

        self.lbl_input_path.configure(text=f"{os.path.basename(file_path)}\n{detail}", text_color=TEXT)
        self._invalidate_cache()
        self._trigger_preview_computation()

    def _browse_folder(self):
        if self.is_processing:
            return
        folder_path = filedialog.askdirectory(title="Choose a folder of photos")
        if not folder_path:
            return

        files = sorted(
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if not files:
            messagebox.showwarning("Empty folder", "No images (.jpg, .png, .webp, .bmp) were found in that folder.")
            return

        self.input_file_path = folder_path
        self.is_folder = True
        self.is_video = False
        self.album_files = files
        self.lbl_input_path.configure(
            text=f"{os.path.basename(folder_path)}\nPhoto album · {len(files)} photos", text_color=TEXT
        )
        self.scrub_frame.grid_remove()
        self._switch_mode_for_photos()

        # First photo is the live preview
        self.current_frame_bgr = imread_safe(files[0])
        self._invalidate_cache()
        self._trigger_preview_computation()

    def _update_scrub_label(self, frame_idx):
        cur_str = time.strftime("%M:%S", time.gmtime(int(frame_idx / self.video_fps)))
        tot_str = time.strftime("%M:%S", time.gmtime(int(self.total_video_frames / self.video_fps)))
        self.lbl_scrub_time.configure(text=f"{cur_str} / {tot_str}")

    def _on_scrub(self, val):
        if not self.is_video or not self.input_file_path:
            return
        self._update_scrub_label(int(val))
        # Debounce: only seek once the user pauses dragging
        if self.scrub_job is not None:
            self.after_cancel(self.scrub_job)
        self.scrub_job = self.after(150, lambda: self._load_frame(int(val)))

    def _load_frame(self, frame_idx):
        self.scrub_job = None
        cap = cv2.VideoCapture(self.input_file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.current_frame_bgr = frame
            self._invalidate_cache()
            self._trigger_preview_computation()

    def _on_model_changed(self, choice):
        self._set_status(f"Switching model: {choice}...")
        self._invalidate_cache()
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _update_fov_visibility(self):
        if self._mode_key() == "vr180":
            self.fov_row.pack(fill="x", padx=12, pady=(2, 8), after=self._fov_anchor)
        else:
            self.fov_row.pack_forget()

    def _on_mode_changed(self, choice):
        self._update_fov_visibility()
        self._render_current_view()

    def _on_profile_changed(self, choice):
        if choice in PROFILES:
            ipd, conv = PROFILES[choice]
            self.slider_ipd.set(ipd)
            self.slider_conv.set(conv)
            self._on_ipd_slide(ipd, refresh=False)
            self._on_conv_slide(conv)

    def _on_ipd_slide(self, val, refresh=True):
        if val < 0.025:
            desc = "Soft"
        elif val <= 0.042:
            desc = "Natural"
        elif val <= 0.058:
            desc = "Strong"
        else:
            desc = "Extreme"
        self.lbl_ipd.configure(text=f"{desc} · {val * 100:.1f}%")
        if refresh:
            self._on_stereo_toggle()

    def _on_conv_slide(self, val, refresh=True):
        self.lbl_conv.configure(text=f"{int(val * 100)}%")
        if refresh:
            self._on_stereo_toggle()

    def _on_fov_slide(self, val, refresh=True):
        self.lbl_fov.configure(text=f"{int(val)}°")
        if refresh:
            self._render_current_view()

    def _on_stereo_toggle(self):
        self._invalidate_stereo_cache()
        self._schedule_preview()

    def _on_appearance_changed(self, choice):
        ctk.set_appearance_mode(choice)

    def _on_settings_change(self):
        if self.current_frame_bgr is not None:
            self._trigger_preview_computation()

    def _on_view_selected(self, label):
        self._select_view(VIEW_KEYS[label])

    def _select_view(self, key):
        self.active_view = key
        self.view_seg.set(VIEW_LABELS[key])
        if self.wiggle_job is not None:
            self.after_cancel(self.wiggle_job)
            self.wiggle_job = None
        self._render_current_view()

    def _on_close(self):
        self._save_settings()
        if self.video_processor and self.is_processing:
            self.video_processor.cancel()
        self.destroy()

    def _save_settings(self):
        data = {
            "model": MODELS.get(self.model_var.get(), DEFAULTS["model"]),
            "mode": self._mode_key(),
            "profile": self.profile_var.get(),
            "ipd": round(self.slider_ipd.get(), 4),
            "conv": round(self.slider_conv.get(), 3),
            "fov": round(self.slider_fov.get(), 1),
            "auto_conv": self.chk_auto_conv.get() == 1,
            "swap": self.chk_swap_eyes.get() == 1,
            "temporal": self.chk_temporal.get() == 1,
            "nvenc": self.chk_nvenc.get() == 1,
            "appearance": self.appearance_seg.get(),
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            print(f"[Settings save error] {exc}")

    # ==========================================
    # Preview pipeline
    # ==========================================
    def _invalidate_cache(self):
        self.cached_depth = None
        self.cached_left = None
        self.cached_right = None

    def _invalidate_stereo_cache(self):
        self.cached_left = None
        self.cached_right = None

    def _update_preview_manual(self):
        if self.is_processing:
            return
        self._invalidate_cache()
        self._trigger_preview_computation()

    def _schedule_preview(self, delay_ms=200):
        """Debounced preview: slider drags only trigger one render after they settle."""
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
        self.preview_job = self.after(delay_ms, self._trigger_preview_computation)

    def _trigger_preview_computation(self):
        self.preview_job = None
        if self.current_frame_bgr is None or self.depth_estimator is None:
            return
        # One preview worker at a time; if busy, re-run once it finishes
        if self.preview_busy:
            self.preview_again = True
            return
        self.preview_busy = True
        threading.Thread(target=self._compute_preview_worker, daemon=True).start()

    def _compute_preview_worker(self):
        try:
            if self.cached_depth is None:
                self.after(0, lambda: self._set_status("Estimating depth..."))
                self.cached_depth = self.depth_estimator.estimate_depth(self.current_frame_bgr)

            if self.cached_left is None or self.cached_right is None:
                self.after(0, lambda: self._set_status("Rendering stereo pair..."))
                p = self._gather_params()
                self.cached_left, self.cached_right = self.stereo_warper.generate_stereo_pair(
                    self.current_frame_bgr, self.cached_depth, ipd_offset=p["ipd_offset"],
                    convergence=p["convergence"], swap_eyes=p["swap_eyes"],
                    auto_convergence=p["auto_convergence"],
                )

            self.after(0, lambda: self._set_status("Preview ready."))
            self.after(0, self._render_current_view)
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Preview error] {err_msg}")
            self.after(0, lambda msg=err_msg: self._set_status(f"Preview error: {msg}"))
        finally:
            self.after(0, self._preview_done)

    def _preview_done(self):
        self.preview_busy = False
        if self.preview_again:
            self.preview_again = False
            self._trigger_preview_computation()

    def _render_current_view(self):
        if self.current_frame_bgr is None:
            return

        if self.wiggle_job is not None:
            self.after_cancel(self.wiggle_job)
            self.wiggle_job = None

        view = self.active_view
        left, right = self.cached_left, self.cached_right
        has_pair = left is not None and right is not None
        out_bgr = None

        if view == "wiggle":
            if has_pair:
                self._step_wiggle()
            return
        elif view == "original":
            out_bgr = self.current_frame_bgr
        elif view == "depth":
            if self.cached_depth is not None:
                out_bgr = self.depth_estimator.depth_to_colormap(self.cached_depth)
        elif view == "left":
            out_bgr = left
        elif view == "right":
            out_bgr = right
        elif view == "sbs" and has_pair:
            out_bgr = self.stereo_warper.create_sbs(left, right, half_sbs=False)
        elif view == "anaglyph" and has_pair:
            out_bgr = self.stereo_warper.create_anaglyph(left, right)
        elif view == "vr180" and has_pair:
            projector = VR180Projector(output_eye_size=(1080, 1080), h_fov_deg=self.slider_fov.get())
            out_bgr = projector.project_vr180_sbs(left, right)

        if out_bgr is not None:
            self._display_bgr_image(out_bgr)

    def _step_wiggle(self):
        if self.active_view != "wiggle" or self.cached_left is None or self.cached_right is None:
            return
        self.wiggle_eye = 1 - self.wiggle_eye
        self._display_bgr_image(self.cached_left if self.wiggle_eye == 0 else self.cached_right)
        self.wiggle_job = self.after(130, self._step_wiggle)

    def _display_bgr_image(self, bgr_img: np.ndarray):
        canvas_w = max(100, self.canvas_frame.winfo_width() - 12)
        canvas_h = max(100, self.canvas_frame.winfo_height() - 12)
        img_h, img_w = bgr_img.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))

        resized = cv2.resize(bgr_img, new_size, interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.lbl_image.configure(image=photo, text="")
        self.lbl_image.image = photo

    # ==========================================
    # Quick sample test (5 seconds)
    # ==========================================
    def _start_quick_test(self):
        if self.is_processing:
            return
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            messagebox.showwarning("No video selected", "Open a video file first to render a quick sample.")
            return
        if not self.is_video:
            messagebox.showinfo("Photos convert instantly",
                                "Photos take ~0.2 s each, so there's no need for a sample. Just press Convert.")
            return
        if self.video_processor is None:
            messagebox.showinfo("Model loading", "The depth model is still loading. Try again in a moment.")
            return

        base_dir, file_name = os.path.split(self.input_file_path)
        name, _ = os.path.splitext(file_name)
        mode = self._mode_key()
        self.output_file_path = os.path.join(base_dir, f"{name}_SAMPLE_5s{SUFFIX_MAP.get(mode, f'_{mode}')}.mp4")

        self._set_busy(True)
        threading.Thread(target=self._quick_test_worker, daemon=True).start()

    def _make_progress_cb(self, label):
        def on_progress(cur, total, fps_val, eta):
            ratio = cur / max(1, total)
            self.after(0, lambda r=ratio: self.progress_bar.set(r))
            self.after(0, lambda c=cur, t=total, p=int(ratio * 100):
                       self._set_status(f"{label}: frame {c}/{t} ({p}%)"))
            self.after(0, lambda f=fps_val, rem=eta: self.stats_label.configure(text=f"{f:.1f} FPS · ETA {rem}"))
        return on_progress

    def _quick_test_worker(self):
        p = self._gather_params()
        temporal = self.chk_temporal.get() == 1
        nvenc = self.chk_nvenc.get() == 1
        try:
            start_frame = int(self.slider_scrub.get())
            success = self.video_processor.process_video(
                self.input_file_path, self.output_file_path,
                use_temporal_filter=temporal, use_nvenc=nvenc,
                progress_callback=self._make_progress_cb("Rendering 5 s sample"),
                start_frame=start_frame, max_frames=int(self.video_fps * 5), **p,
            )
            if success:
                out = self.output_file_path
                self.after(0, lambda: self._offer_open_folder(
                    "Sample ready",
                    f"Your 5-second 3D sample is ready:\n{out}\n\nCheck it in your headset or player.",
                    os.path.dirname(out)))
            else:
                self.after(0, lambda: self._set_status("Sample cancelled."))
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Quick test error] {err_msg}")
            self.after(0, lambda msg=err_msg: messagebox.showerror("Sample failed", f"Could not render the sample:\n{msg}"))
        finally:
            self.after(0, lambda: self._set_busy(False))
            self.after(0, lambda: self._set_status("Ready."))

    # ==========================================
    # Conversion
    # ==========================================
    def _start_conversion(self):
        if self.is_processing:
            return
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            messagebox.showwarning("Nothing to convert", "Open a video, photo, or photo folder first.")
            return
        if self.video_processor is None:
            messagebox.showinfo("Model loading", "The depth model is still loading. Try again in a moment.")
            return

        if self.is_folder:
            out_path = filedialog.askdirectory(title="Choose where to save the 3D photos",
                                               initialdir=self.input_file_path)
        else:
            base_dir, file_name = os.path.split(self.input_file_path)
            name, ext = os.path.splitext(file_name)
            mode = self._mode_key()
            out_path = filedialog.asksaveasfilename(
                title="Save output as",
                initialdir=base_dir,
                initialfile=f"{name}{SUFFIX_MAP.get(mode, f'_{mode}')}{ext}",
                defaultextension=".mp4" if self.is_video else ".jpg",
                filetypes=[("Video", "*.mp4")] if self.is_video else [("Image", "*.jpg *.png")],
            )
        if not out_path:
            return

        self.output_file_path = out_path
        self._set_busy(True)
        threading.Thread(target=self._conversion_worker, daemon=True).start()

    def _cancel_conversion(self):
        self.is_processing = False
        if self.video_processor:
            self.video_processor.cancel()
        self._set_status("Cancelling...")

    def _conversion_worker(self):
        p = self._gather_params()
        temporal = self.chk_temporal.get() == 1
        nvenc = self.chk_nvenc.get() == 1
        out = self.output_file_path

        try:
            if self.is_folder:
                self._convert_album(p, out)
            elif not self.is_video:
                self.after(0, lambda: self._set_status("Converting image..."))
                self.video_processor.process_image(self.input_file_path, out, **p)
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, lambda: self._offer_open_folder(
                    "Done", f"Your 3D image is ready:\n{out}", os.path.dirname(out)))
            else:
                success = self.video_processor.process_video(
                    self.input_file_path, out,
                    use_temporal_filter=temporal, use_nvenc=nvenc,
                    progress_callback=self._make_progress_cb("Converting"), **p,
                )
                if success:
                    self.after(0, lambda: self._offer_open_folder(
                        "Done", f"Your 3D video is ready:\n{out}", os.path.dirname(out)))
                else:
                    self.after(0, lambda: self._set_status("Conversion cancelled."))
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Conversion error] {err_msg}")
            self.after(0, lambda msg=err_msg: messagebox.showerror("Conversion failed", f"Could not convert:\n{msg}"))
        finally:
            self.after(0, lambda: self._set_busy(False))
            self.after(0, lambda: self._set_status("Ready."))

    def _convert_album(self, p, out_dir):
        total = len(self.album_files)
        os.makedirs(out_dir, exist_ok=True)
        suffix = SUFFIX_MAP.get(p["mode"], f"_{p['mode']}")
        start_time = time.time()
        success_count = 0

        for idx, in_img_path in enumerate(self.album_files):
            if not self.is_processing:
                break

            root, ext = os.path.splitext(os.path.basename(in_img_path))
            clean_root = root if root.endswith(suffix) else f"{root}{suffix}"
            out_img_path = os.path.join(out_dir, f"{clean_root}{ext}")

            ratio = idx / max(1, total)
            self.after(0, lambda r=ratio: self.progress_bar.set(r))
            self.after(0, lambda i=idx + 1, pct=int(ratio * 100):
                       self._set_status(f"Converting album: photo {i}/{total} ({pct}%)"))

            try:
                self.video_processor.process_image(in_img_path, out_img_path, **p)
                success_count += 1
            except Exception as img_err:
                print(f"[Album error on {os.path.basename(in_img_path)}]: {img_err}")

            rate = (idx + 1) / max(0.001, time.time() - start_time)
            eta = time.strftime("%M:%S", time.gmtime(int((total - idx - 1) / max(0.001, rate))))
            self.after(0, lambda f=rate, rem=eta: self.stats_label.configure(text=f"{f:.1f} photos/s · ETA {rem}"))

        if self.is_processing:
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self._offer_open_folder(
                "Done", f"Converted {success_count} of {total} photos into:\n{out_dir}", out_dir))
        else:
            self.after(0, lambda: self._set_status("Album conversion cancelled."))


def main():
    app = VR3DStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
