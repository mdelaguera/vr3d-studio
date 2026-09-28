"""
Any2VR desktop app. Built with CustomTkinter on top of `engine` (all conversion logic lives there).

Modules: theme (look), prefs (settings + recents), system_check (setup advice), dialogs (welcome, fixes).
This file wires them into the main window: sidebar controls, live preview, jobs.
"""

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine import (CATALOG, DEFAULT_MODEL, Converter, OutputFormat, OutputSettings, StereoSettings,
                    compose, is_image, is_video, load_model, output_path_for)
from engine.media import VideoReader, read_image
from gui import prefs, system_check, theme as t
from gui.dialogs import FixDialog, WelcomeDialog
from gui.prefs import MODELS, MODES, PROFILES, label_for

# Drag & drop is optional: the app still works (browse buttons only) without tkinterdnd2
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_BASES = (TkinterDnD.DnDWrapper,)
except ImportError:
    TkinterDnD = None
    DND_BASES = ()

# Preview runs on a downscaled copy (the canvas is smaller anyway); exports use full resolution.
# ponytail: fixed cap, make it follow canvas size if 4K monitors look soft
PREVIEW_MAX_SIDE = 960

# (internal key, button label)
VIEWS = [
    ("wiggle", "Wiggle 3D"), ("sbs", "SBS"), ("vr180", "VR180"), ("anaglyph", "Anaglyph"),
    ("depth", "Depth"), ("left", "Left eye"), ("right", "Right eye"), ("original", "Original"),
]
PREVIEW_FORMATS = {"sbs": OutputFormat.SBS_FULL, "vr180": OutputFormat.VR180, "anaglyph": OutputFormat.ANAGLYPH}
VIEW_LABELS = dict(VIEWS)
VIEW_KEYS = {v: k for k, v in VIEWS}

WATCH_TIPS = {
    "vr180": "Copy it to your headset and open it in any VR video player. The _180_SBS name tells it the format.",
    "sbs_full": "Play it on a 3D TV/monitor in side-by-side mode, or in a VR player's 3D SBS mode.",
    "sbs_half": "Play it on a 3D TV in half side-by-side mode.",
    "anaglyph": "Put on red/cyan glasses and open it in any player.",
    "depth_only": "",
}


def open_path(path):
    try:
        os.startfile(path)
    except (OSError, AttributeError):
        subprocess.Popen(["xdg-open" if sys.platform.startswith("linux") else "open", path])


def show_in_folder(path):
    if sys.platform == "win32" and os.path.isfile(path):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        open_path(path if os.path.isdir(path) else os.path.dirname(path))


class Any2VRApp(ctk.CTk, *DND_BASES):
    def __init__(self):
        self.settings = prefs.load()
        ctk.set_appearance_mode(self.settings["appearance"])
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.dnd_enabled = False
        if TkinterDnD is not None:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self.dnd_enabled = True
            except Exception as exc:
                print(f"[Drag & drop unavailable] {exc}")

        self.title("Any2VR - Turn any photo or video into 3D & VR")
        self.geometry("1280x820")
        self.minsize(980, 660)
        self.configure(fg_color=t.MAIN_BG)

        self.converter = None            # engine.Converter, set once the depth model loads
        self.cancel_event = threading.Event()
        self.report = system_check.SystemReport()

        # Source / preview state
        self.input_file_path = ""
        self.is_video = False
        self.is_folder = False
        self.total_video_frames = 0
        self.video_fps = 30.0
        self.video_cap = None
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
        self.resize_job = None
        self.preview_busy = False
        # Bumped whenever inputs change; stale preview results are discarded on commit
        self.depth_gen = 0
        self.stereo_gen = 0

        self._build_ui()
        self._bind_shortcuts()
        if self.dnd_enabled:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._system_check_worker, daemon=True).start()
        self._load_model_async(MODELS.get(self.model_var.get(), DEFAULT_MODEL))

    # ==========================================
    # Startup: setup check + model loading (background threads)
    # ==========================================
    def _system_check_worker(self):
        report = system_check.check()
        self.after(0, lambda: self._on_system_report(report))

    def _on_system_report(self, report):
        self.report = report
        ai = ("⚡ " if report.has_gpu else "") + ("CPU" if report.device_label.startswith("CPU") else report.device_label)
        self.hw_label.configure(text=f"AI: {ai}" + (f"  ·  Video: {report.encoder}" if report.encoder else ""),
                                text_color=t.GO if report.has_gpu else t.WARN)
        self._update_model_hint()
        if report.issues:
            top = next((i for i in report.issues if i.blocking), report.issues[0])
            self.banner_title.configure(text=top.title)
            self.banner_detail.configure(text=top.detail)
            self.banner_fix.configure(command=lambda: FixDialog(self, top))
            self.banner_fix.grid() if top.fix_command else self.banner_fix.grid_remove()
            self.banner.grid()
        if not self.settings["welcomed"]:
            self.settings["welcomed"] = True
            WelcomeDialog(self, report, on_fix=lambda issue: FixDialog(self, issue))

    def _load_model_async(self, model_id):
        info = CATALOG[model_id]
        self._set_status(f"Loading {info.name} (first use downloads ~{info.download_mb} MB, then it's cached)...")
        threading.Thread(target=self._load_model_worker, args=(model_id,), daemon=True).start()

    def _load_model_worker(self, model_id):
        try:
            converter = Converter(load_model(model_id))
            self.after(0, lambda: self._on_model_ready(converter))
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Model load error] {err_msg}")
            self.after(0, lambda: self._set_status(f"Couldn't load the model: {err_msg}"))

    def _on_model_ready(self, converter):
        self.converter = converter
        self._set_status("Ready. Drop a photo or video to begin." if self.current_frame_bgr is None else "Ready.")
        self._invalidate_cache()
        self._trigger_preview_computation()

    # ==========================================
    # UI construction
    # ==========================================
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        s = self.settings
        sidebar = ctk.CTkFrame(self, width=330, corner_radius=0, fg_color=t.SIDEBAR)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(1, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        # --- Header ---
        header = ctk.CTkFrame(sidebar, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        ctk.CTkLabel(header, text="Any2VR", font=t.font(22, True), text_color=t.TEXT).pack(anchor="w")
        ctk.CTkLabel(header, text="Private, local 2D → 3D & VR", font=t.font(12), text_color=t.MUTED).pack(anchor="w")
        self.hw_label = ctk.CTkLabel(header, text="Checking hardware...", font=t.font(11, True), text_color=t.MUTED,
                                     fg_color=t.CARD, corner_radius=10, height=24)
        self.hw_label.pack(anchor="w", pady=(8, 0), ipadx=8)

        body = ctk.CTkScrollableFrame(sidebar, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=4)

        # 1 Source
        src = t.card(body, "1  Source")
        row = ctk.CTkFrame(src, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(2, 4))
        row.grid_columnconfigure((0, 1), weight=1)
        t.button(row, "Open file", self._browse_input, kind="primary").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        t.button(row, "Open folder", self._browse_folder).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.lbl_input_path = t.hint(src, "No file yet · drag one in or press Ctrl+O")
        self.lbl_input_path.pack_configure(pady=(0, 10))

        # 2 Model & format
        fmt = t.card(body, "2  Model & format")
        t.field_label(fmt, "Depth model")
        self.model_var = ctk.StringVar(value=label_for(MODELS, s["model"]))
        t.option_menu(fmt, list(MODELS), self.model_var, self._on_model_changed)
        self.model_hint = t.hint(fmt)
        t.field_label(fmt, "Output format")
        self.mode_var = ctk.StringVar(value=label_for(MODES, s["mode"]))
        t.option_menu(fmt, list(MODES), self.mode_var, self._on_mode_changed)
        ctk.CTkFrame(fmt, height=6, fg_color="transparent").pack()
        self._update_model_hint()

        # 3 3D effect
        fx = t.card(body, "3  3D effect")
        t.field_label(fx, "Preset")
        self.profile_var = ctk.StringVar(value=s["profile"])
        t.option_menu(fx, list(PROFILES), self.profile_var, self._on_profile_changed)
        ctk.CTkFrame(fx, height=4, fg_color="transparent").pack()
        _, self.lbl_ipd, self.slider_ipd = t.slider_row(fx, "Depth strength", 0.010, 0.090, 80, s["ipd"], self._on_ipd_slide)
        _, self.lbl_conv, self.slider_conv = t.slider_row(fx, "Focus plane", 0.1, 0.9, 80, s["conv"], self._on_conv_slide)
        self.chk_auto_conv = t.switch(fx, "Auto focus on subject", s["auto_conv"], self._on_stereo_toggle)
        self.fov_row, self.lbl_fov, self.slider_fov = t.slider_row(fx, "VR field of view", 80.0, 150.0, 70, s["fov"], self._on_fov_slide)
        ctk.CTkFrame(fx, height=4, fg_color="transparent").pack()
        self._fov_anchor = self.chk_auto_conv

        # 4 Options
        opts = t.card(body, "4  Options")
        self.chk_swap_eyes = t.switch(opts, "Swap left / right eyes", s["swap"], self._on_stereo_toggle)
        self.chk_temporal = t.switch(opts, "Anti-flicker smoothing (video)", s["temporal"])
        self.chk_nvenc = t.switch(opts, "GPU video encoding (NVIDIA / AMD / Intel)", s["nvenc"])
        ctk.CTkFrame(opts, height=6, fg_color="transparent").pack()

        # Recent outputs
        self.recent_card = t.card(body, "Recent")
        self.recent_list = ctk.CTkFrame(self.recent_card, fg_color="transparent")
        self.recent_list.pack(fill="x", padx=8, pady=(0, 8))
        self._render_recent()

        # Appearance
        look = t.card(body, "Appearance")
        self.appearance_seg = ctk.CTkSegmentedButton(look, values=["Dark", "Light", "System"], command=ctk.set_appearance_mode,
                                                     selected_color=t.ACCENT, selected_hover_color=t.ACCENT_HOVER)
        self.appearance_seg.set(s["appearance"])
        self.appearance_seg.pack(fill="x", padx=12, pady=(2, 12))

        self._on_ipd_slide(s["ipd"], refresh=False)
        self._on_conv_slide(s["conv"], refresh=False)
        self._on_fov_slide(s["fov"], refresh=False)
        self._update_fov_visibility()

        # --- Sticky action bar (always visible) ---
        actions = ctk.CTkFrame(sidebar, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 16))
        actions.grid_columnconfigure((0, 1), weight=1)
        self.btn_preview = t.button(actions, "Refresh preview", self._update_preview_manual)
        self.btn_preview.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
        self.btn_quick_test = t.button(actions, "5 s sample", self._start_quick_test)
        self.btn_quick_test.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))
        self.btn_start = t.button(actions, "Convert  (Ctrl+Enter)", self._start_conversion, kind="go", height=42,
                                  font=t.font(14, True))
        self.btn_start.grid(row=1, column=0, columnspan=2, sticky="ew")
        # Cancel replaces Convert while busy (same grid cell)
        self.btn_cancel = t.button(actions, "Cancel  (Esc)", self._cancel_conversion, kind="danger", height=42,
                                   font=t.font(14, True))

    def _build_main(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        # Setup banner (only when the system check finds something)
        self.banner = ctk.CTkFrame(main, fg_color=t.CARD, corner_radius=12, border_width=1, border_color=t.WARN)
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.banner.grid_columnconfigure(0, weight=1)
        self.banner_title = ctk.CTkLabel(self.banner, text="", font=t.font(13, True), text_color=t.WARN)
        self.banner_title.grid(row=0, column=0, sticky="w", padx=14, pady=(8, 0))
        self.banner_detail = ctk.CTkLabel(self.banner, text="", font=t.font(12), text_color=t.MUTED, justify="left",
                                          wraplength=640)
        self.banner_detail.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 8))
        btns = ctk.CTkFrame(self.banner, fg_color="transparent")
        btns.grid(row=0, column=1, rowspan=2, padx=10)
        self.banner_fix = t.button(btns, "How to fix", None, kind="primary", width=100)
        self.banner_fix.grid(row=0, column=0, padx=4)
        t.button(btns, "✕", self.banner.grid_remove, width=32).grid(row=0, column=1, padx=4)
        self.banner.grid_remove()

        # View switcher
        self.view_seg = ctk.CTkSegmentedButton(
            main, values=[label for _, label in VIEWS], command=lambda label: self._select_view(VIEW_KEYS[label]),
            height=32, selected_color=t.ACCENT, selected_hover_color=t.ACCENT_HOVER, font=t.font(12, True))
        self.view_seg.set(VIEW_LABELS[self.active_view])
        self.view_seg.grid(row=1, column=0, sticky="w", pady=(0, 10))

        # Preview canvas
        self.canvas_frame = ctk.CTkFrame(main, fg_color=t.PREVIEW_BG, corner_radius=12)
        self.canvas_frame.grid(row=2, column=0, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.lbl_image = tk.Label(
            self.canvas_frame, bg=t.PREVIEW_BG, fg="#8B90A3", font=("Segoe UI", 13),
            text="Drop a photo, video, or folder here\n\n"
                 "Ctrl+O  open file     ·     Ctrl+Shift+O  open folder\n"
                 "1-8  switch view     ·     F5  refresh preview")
        self.lbl_image.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.canvas_frame.bind("<Configure>", self._on_canvas_resize)

        # Video timeline scrubber (videos only)
        self.scrub_frame = ctk.CTkFrame(main, fg_color="transparent")
        self.scrub_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.scrub_frame.grid_columnconfigure(1, weight=1)
        self.lbl_scrub_time = ctk.CTkLabel(self.scrub_frame, text="00:00 / 00:00", width=96, font=t.font(12),
                                           text_color=t.MUTED)
        self.lbl_scrub_time.grid(row=0, column=0, padx=(0, 10))
        self.slider_scrub = ctk.CTkSlider(self.scrub_frame, from_=0, to=100, command=self._on_scrub,
                                          button_color=t.ACCENT, button_hover_color=t.ACCENT_HOVER, progress_color=t.ACCENT)
        self.slider_scrub.set(0)
        self.slider_scrub.grid(row=0, column=1, sticky="ew")
        self.scrub_frame.grid_remove()

        # Status bar with progress, and result actions once a job finishes
        bar = ctk.CTkFrame(main, fg_color=t.CARD, corner_radius=12)
        bar.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(bar, text="Starting...", font=t.font(12), text_color=t.TEXT, justify="left")
        self.status_label.grid(row=0, column=0, sticky="w", padx=14, pady=(8, 2))
        self.stats_label = ctk.CTkLabel(bar, text="", font=t.font(12, True), text_color=t.ACCENT)
        self.stats_label.grid(row=0, column=1, sticky="e", padx=14, pady=(8, 2))
        self.result_actions = ctk.CTkFrame(bar, fg_color="transparent")
        self.result_actions.grid(row=0, column=2, rowspan=2, sticky="e", padx=(0, 10))
        self.btn_open_result = t.button(self.result_actions, "Open", None, kind="primary", width=80)
        self.btn_open_result.pack(side="left", padx=4)
        self.btn_show_result = t.button(self.result_actions, "Show in folder", None, width=110)
        self.btn_show_result.pack(side="left", padx=4)
        self.result_actions.grid_remove()
        self.tip_label = ctk.CTkLabel(bar, text="", font=t.font(11), text_color=t.MUTED, justify="left")
        self.tip_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=14)
        self.progress_bar = ctk.CTkProgressBar(bar, height=6, progress_color=t.ACCENT, fg_color=t.SUBTLE)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 10))

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self._browse_input())
        self.bind("<Control-O>", lambda e: self._browse_folder())  # Ctrl+Shift+O
        self.bind("<Control-Return>", lambda e: self._start_conversion())
        self.bind("<Escape>", lambda e: self._cancel_conversion() if self.is_processing else None)
        self.bind("<F5>", lambda e: self._update_preview_manual())
        for i, (key, _) in enumerate(VIEWS, start=1):
            self.bind(str(i), lambda e, k=key: self._select_view(k))

    # ==========================================
    # Small helpers
    # ==========================================
    def _set_status(self, text):
        self.status_label.configure(text=text)

    def _mode_key(self):
        return MODES.get(self.mode_var.get(), "vr180")

    def _update_model_hint(self):
        info = CATALOG[MODELS.get(self.model_var.get(), DEFAULT_MODEL)]
        slow_on_cpu = info.speed == "slow" and not self.report.has_gpu and self.report.device_label.startswith("CPU")
        text = f"{info.blurb} ~{info.download_mb} MB, runs privately on this PC."
        if slow_on_cpu:
            text += " Very slow without a GPU: fine for a few photos, not for video."
        self.model_hint.configure(text=text, text_color=t.WARN if slow_on_cpu else t.MUTED)

    def _set_busy(self, busy):
        self.is_processing = busy
        state = "disabled" if busy else "normal"
        self.btn_quick_test.configure(state=state)
        self.btn_preview.configure(state=state)
        if busy:
            self.result_actions.grid_remove()
            self.tip_label.configure(text="")
            self.btn_start.grid_remove()
            self.btn_cancel.grid(row=1, column=0, columnspan=2, sticky="ew")
            self.progress_bar.set(0.0)
        else:
            self.btn_cancel.grid_remove()
            self.btn_start.grid()

    # Widget -> engine settings. UI thread only: workers get these snapshots, never widgets.
    def _stereo_settings(self) -> StereoSettings:
        return StereoSettings(strength=self.slider_ipd.get(), focus=self.slider_conv.get(),
                              auto_focus=self.chk_auto_conv.get() == 1, swap_eyes=self.chk_swap_eyes.get() == 1)

    def _output_settings(self, eye_size: int = 1920) -> OutputSettings:
        return OutputSettings(OutputFormat(self._mode_key()), vr_fov=self.slider_fov.get(), vr_eye_size=eye_size)

    def _switch_mode_for_photos(self):
        """VR180 is meant for video; photos look best as Full SBS."""
        if self._mode_key() == "vr180":
            sbs_label = label_for(MODES, "sbs_full")
            self.mode_var.set(sbs_label)
            self._on_mode_changed(sbs_label)

    def _render_recent(self):
        for child in self.recent_list.winfo_children():
            child.destroy()
        recent = prefs.prune_recent(self.settings["recent"])
        self.settings["recent"] = recent
        if not recent:
            ctk.CTkLabel(self.recent_list, text="Your converted files will show up here.", font=t.font(11),
                         text_color=t.MUTED).pack(anchor="w", padx=4)
            return
        icons = {"video": "🎞", "image": "🖼", "folder": "📁"}
        for r in recent:
            row = ctk.CTkFrame(self.recent_list, fg_color="transparent")
            row.pack(fill="x", pady=1)
            name = os.path.basename(r["path"].rstrip("/\\"))
            label = f"{icons.get(r['kind'], '•')}  {name if len(name) <= 30 else name[:28] + '…'}"
            ctk.CTkButton(row, text=label, anchor="w", height=26, fg_color="transparent", hover_color=t.SUBTLE_HOVER,
                          text_color=t.TEXT, font=t.font(12), command=lambda p=r["path"]: open_path(p)
                          ).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(row, text="📂", width=30, height=26, fg_color="transparent", hover_color=t.SUBTLE_HOVER,
                          text_color=t.TEXT, command=lambda p=r["path"]: show_in_folder(p)).pack(side="right")

    def _show_result(self, message, path, kind, tip=""):
        """Non-blocking 'done' state: message, Open / Show in folder buttons, and a how-to-watch tip."""
        self._set_status(f"✓ {message}")
        self.tip_label.configure(text=tip)
        self.btn_open_result.configure(command=lambda: open_path(path))
        self.btn_show_result.configure(command=lambda: show_in_folder(path))
        self.result_actions.grid()
        self.settings["recent"] = prefs.add_recent(self.settings["recent"], path, kind)
        self._render_recent()
        prefs.save(self._collect_settings())

    # ==========================================
    # Source loading
    # ==========================================
    def _browse_input(self):
        if self.is_processing:
            return
        file_path = filedialog.askopenfilename(
            title="Choose a video or photo",
            filetypes=[
                ("Media files", "*.mp4 *.mkv *.mov *.avi *.webm *.jpg *.jpeg *.png *.webp *.bmp"),
                ("Videos", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All files", "*.*"),
            ])
        if file_path:
            self._open_file(file_path)

    def _browse_folder(self):
        if self.is_processing:
            return
        folder_path = filedialog.askdirectory(title="Choose a folder of photos")
        if folder_path:
            self._open_folder(folder_path)

    def _on_drop(self, event):
        if self.is_processing:
            return
        paths = self.tk.splitlist(event.data)  # handles {paths with spaces}
        if not paths:
            return
        path = paths[0]
        if os.path.isdir(path):
            self._open_folder(path)
        elif is_video(path) or is_image(path):
            self._open_file(path)
        else:
            messagebox.showwarning("Unsupported file", f"Drop a video, image, or folder of images.\n\n{path}")

    def _close_video(self):
        if self.video_cap is not None:
            self.video_cap.close()
            self.video_cap = None

    def _open_file(self, file_path):
        video = is_video(file_path)
        # Read first; only switch the app to this file if it actually opens
        try:
            if video:
                reader = VideoReader(file_path)
                frame = reader.read_at(0)
                if frame is None:
                    reader.close()
                    raise ValueError(f"Could not read frames from: {file_path}")
            else:
                frame = read_image(file_path)
        except ValueError as exc:
            messagebox.showerror("Can't open file", str(exc))
            return

        self._close_video()
        self.input_file_path = file_path
        self.is_folder = False
        self.is_video = video
        if video:
            self.video_cap = reader  # kept open so scrubbing doesn't reopen the file each time
            self.total_video_frames = reader.frame_count
            self.video_fps = reader.fps
            self.slider_scrub.configure(to=max(1, self.total_video_frames - 1))
            self.slider_scrub.set(0)
            self._update_scrub_label(0)
            self.scrub_frame.grid()
            duration = time.strftime("%M:%S", time.gmtime(int(self.total_video_frames / self.video_fps)))
            detail = f"Video · {duration} · {self.video_fps:.0f} fps · {reader.size[0]}×{reader.size[1]}"
        else:
            self.scrub_frame.grid_remove()
            self._switch_mode_for_photos()
            detail = f"Photo · {frame.shape[1]}×{frame.shape[0]}"
        self.lbl_input_path.configure(text=f"{os.path.basename(file_path)}\n{detail}", text_color=t.TEXT)
        self._set_source_frame(frame)

    def _open_folder(self, folder_path):
        files = sorted(os.path.join(folder_path, f) for f in os.listdir(folder_path) if is_image(f))
        if not files:
            messagebox.showwarning("Empty folder", "No images (.jpg, .png, .webp, .bmp) were found in that folder.")
            return

        def try_read(path):
            try:
                return read_image(path)
            except ValueError:
                return None

        first = next((img for img in map(try_read, files[:10]) if img is not None), None)
        if first is None:
            messagebox.showwarning("Unreadable images", "Couldn't read the first images in that folder.")
            return

        self._close_video()
        self.input_file_path = folder_path
        self.is_folder = True
        self.is_video = False
        count = f"{len(files)} photo{'' if len(files) == 1 else 's'}"
        self.lbl_input_path.configure(text=f"{os.path.basename(folder_path)}\nPhoto album · {count}", text_color=t.TEXT)
        self.scrub_frame.grid_remove()
        self._switch_mode_for_photos()
        self._set_source_frame(first)

    def _set_source_frame(self, frame):
        """New input frame: downscale for preview, show it immediately, then recompute 3D."""
        h, w = frame.shape[:2]
        scale = PREVIEW_MAX_SIDE / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.current_frame_bgr = frame
        self._invalidate_cache()
        # Show the raw frame right away so the picture always matches the time bar
        self._stop_wiggle()
        self._display_bgr_image(frame)
        self._trigger_preview_computation()

    def _update_scrub_label(self, frame_idx):
        cur = time.strftime("%M:%S", time.gmtime(int(frame_idx / self.video_fps)))
        tot = time.strftime("%M:%S", time.gmtime(int(self.total_video_frames / self.video_fps)))
        self.lbl_scrub_time.configure(text=f"{cur} / {tot}")

    def _on_scrub(self, val):
        if not self.is_video:
            return
        self._update_scrub_label(int(val))
        # Debounce: only seek once the user pauses dragging
        if self.scrub_job is not None:
            self.after_cancel(self.scrub_job)
        self.scrub_job = self.after(150, lambda: self._load_frame(int(val)))

    def _load_frame(self, frame_idx):
        self.scrub_job = None
        if self.video_cap is None:
            return
        frame = self.video_cap.read_at(frame_idx)
        if frame is not None:
            self._set_source_frame(frame)

    # ==========================================
    # Settings handlers
    # ==========================================
    def _on_model_changed(self, choice):
        self._update_model_hint()
        self._invalidate_cache()
        self._load_model_async(MODELS.get(choice, DEFAULT_MODEL))

    def _update_fov_visibility(self):
        if self._mode_key() == "vr180":
            self.fov_row.pack(fill="x", padx=12, pady=(2, 8), after=self._fov_anchor)
        else:
            self.fov_row.pack_forget()

    def _on_mode_changed(self, _choice):
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
        desc = "Soft" if val < 0.025 else "Natural" if val <= 0.042 else "Strong" if val <= 0.058 else "Extreme"
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

    def _select_view(self, key):
        self.active_view = key
        self.view_seg.set(VIEW_LABELS[key])
        self._render_current_view()

    def _collect_settings(self) -> dict:
        return {
            **self.settings,
            "model": MODELS.get(self.model_var.get(), DEFAULT_MODEL),
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

    def _on_close(self):
        prefs.save(self._collect_settings())
        self._close_video()
        self.cancel_event.set()
        self.destroy()

    # ==========================================
    # Preview pipeline
    # ==========================================
    def _invalidate_cache(self):
        self.depth_gen += 1
        self.cached_depth = None
        self._invalidate_stereo_cache()

    def _invalidate_stereo_cache(self):
        self.stereo_gen += 1
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
        if self.current_frame_bgr is None or self.converter is None:
            return
        # One worker at a time; _preview_done re-triggers if inputs changed meanwhile
        if self.preview_busy or (self.cached_depth is not None and self.cached_left is not None):
            return
        self.preview_busy = True
        # Snapshot every input on the UI thread; the worker never touches Tk widgets or caches
        job = dict(converter=self.converter, frame=self.current_frame_bgr, depth=self.cached_depth,
                   depth_gen=self.depth_gen, stereo_gen=self.stereo_gen, stereo=self._stereo_settings())
        threading.Thread(target=self._compute_preview_worker, args=(job,), daemon=True).start()

    def _compute_preview_worker(self, job):
        result = None
        try:
            t0 = time.time()
            conv, depth = job["converter"], job["depth"]
            if depth is None:
                self.after(0, lambda: self._set_status("Estimating depth..."))
                depth = conv.depth_model.estimate(job["frame"])
            left, right = conv.stereo_pair(job["frame"], depth, job["stereo"])
            result = (depth, left, right, time.time() - t0)
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Preview error] {err_msg}")
            self.after(0, lambda: self._set_status(f"Preview error: {err_msg}"))
        finally:
            self.after(0, lambda: self._preview_done(job, result))

    def _preview_done(self, job, result):
        """Runs on the UI thread. Commits only results whose inputs are still current."""
        self.preview_busy = False
        if result is not None:
            depth, left, right, secs = result
            if job["depth_gen"] == self.depth_gen:
                self.cached_depth = depth
                if job["stereo_gen"] == self.stereo_gen:
                    self.cached_left, self.cached_right = left, right
                    if not self.is_processing:
                        self._set_status(f"Preview ready ({secs:.1f} s).")
                    self._render_current_view()
        # Frame or settings changed while we worked: go again with the fresh inputs
        if (job["depth_gen"], job["stereo_gen"]) != (self.depth_gen, self.stereo_gen):
            self._trigger_preview_computation()

    def _render_current_view(self):
        self.resize_job = None
        if self.current_frame_bgr is None:
            return
        self._stop_wiggle()
        view, left, right = self.active_view, self.cached_left, self.cached_right
        has_pair = left is not None and right is not None
        out = None
        if view == "wiggle":
            if has_pair:
                self._step_wiggle()
            return
        elif view == "original":
            out = self.current_frame_bgr
        elif view == "left":
            out = left
        elif view == "right":
            out = right
        elif view == "depth" and self.cached_depth is not None:
            out = compose(OutputFormat.DEPTH, left, right, self.cached_depth, None)
        elif view in PREVIEW_FORMATS and has_pair:
            out = compose(PREVIEW_FORMATS[view], left, right, self.cached_depth, self._output_settings(eye_size=1080))
        if out is not None:
            self._display_bgr_image(out)

    def _stop_wiggle(self):
        if self.wiggle_job is not None:
            self.after_cancel(self.wiggle_job)
            self.wiggle_job = None

    def _on_canvas_resize(self, _evt=None):
        # Window drags fire dozens of events; redraw once when resizing pauses
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(80, self._render_current_view)

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
        size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        resized = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.lbl_image.configure(image=photo, text="")
        self.lbl_image.image = photo

    # ==========================================
    # Jobs: 5 s sample and full conversion (all work runs through engine.Converter)
    # ==========================================
    def _check_ready(self) -> bool:
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            messagebox.showwarning("Nothing to convert", "Open a video, photo, or photo folder first.")
            return False
        if self.converter is None:
            messagebox.showinfo("Model loading", "The depth model is still loading. Try again in a moment.")
            return False
        if self.is_video and any(i.blocking for i in self.report.issues):
            FixDialog(self, next(i for i in self.report.issues if i.blocking))
            return False
        return True

    def _start_job(self, kind: str, dst: str, **extra):
        """Snapshot settings on the UI thread, then run the job on a worker thread."""
        job = dict(kind=kind, src=self.input_file_path, dst=dst, converter=self.converter,
                   stereo=self._stereo_settings(), output=self._output_settings(), mode=self._mode_key(),
                   smooth=self.chk_temporal.get() == 1, nvenc=self.chk_nvenc.get() == 1, **extra)
        self.cancel_event.clear()
        self._set_busy(True)
        threading.Thread(target=self._job_worker, args=(job,), daemon=True).start()

    def _start_quick_test(self):
        if self.is_processing or not self._check_ready():
            return
        if not self.is_video:
            messagebox.showinfo("Photos convert quickly", "There's no need for a sample with photos. Just press Convert.")
            return
        root, _ = os.path.splitext(output_path_for(self.input_file_path, OutputFormat(self._mode_key())))
        self._start_job("sample", root + "_SAMPLE_5s.mp4",
                        start_frame=int(self.slider_scrub.get()), max_frames=int(self.video_fps * 5))

    def _start_conversion(self):
        if self.is_processing or not self._check_ready():
            return
        if self.is_folder:
            dst = filedialog.askdirectory(title="Choose where to save the 3D photos", initialdir=self.input_file_path)
            kind = "folder"
        else:
            suggested = output_path_for(self.input_file_path, OutputFormat(self._mode_key()),
                                        ext=".mp4" if self.is_video else None)
            dst = filedialog.asksaveasfilename(
                title="Save output as", initialdir=os.path.dirname(suggested), initialfile=os.path.basename(suggested),
                defaultextension=".mp4" if self.is_video else ".jpg",
                filetypes=[("Video", "*.mp4")] if self.is_video else [("Image", "*.jpg *.png")])
            kind = "video" if self.is_video else "image"
        if dst:
            self._start_job(kind, dst)

    def _cancel_conversion(self):
        self.cancel_event.set()
        self._set_status("Cancelling...")

    def _progress_cb(self, label: str, unit: str):
        def on_progress(p):
            eta = time.strftime("%M:%S", time.gmtime(int(p.eta_seconds)))
            self.after(0, lambda: self.progress_bar.set(p.fraction))
            self.after(0, lambda: self._set_status(f"{label}: {unit} {p.done}/{p.total} ({p.fraction:.0%})"))
            self.after(0, lambda: self.stats_label.configure(text=f"{p.rate:.1f} {unit}s/s · ETA {eta}"))
        return on_progress

    def _job_worker(self, job):
        conv, src, dst, kind = job["converter"], job["src"], job["dst"], job["kind"]
        stereo, output, cancel = job["stereo"], job["output"], self.cancel_event
        tip = WATCH_TIPS.get(job["mode"], "")
        result = None  # (message, path, recent-kind)
        try:
            if kind == "folder":
                ok, bad = conv.convert_folder(src, dst, stereo, output, cancel=cancel,
                                              progress=self._progress_cb("Converting album", "photo"))
                if not cancel.is_set():
                    skipped = f" · {len(bad)} unreadable file{'s' if len(bad) != 1 else ''} skipped" if bad else ""
                    result = (f"Converted {len(ok)} photos{skipped}", dst, "folder")
            elif kind == "image":
                self.after(0, lambda: self._set_status("Converting image..."))
                conv.convert_image(src, dst, stereo, output)
                result = (f"Saved {os.path.basename(dst)}", dst, "image")
            else:
                sample = kind == "sample"
                done = conv.convert_video(
                    src, dst, stereo, output, smooth=job["smooth"], prefer_hardware=job["nvenc"], cancel=cancel,
                    start_frame=job.get("start_frame", 0), max_frames=job.get("max_frames"),
                    progress=self._progress_cb("Rendering 5 s sample" if sample else "Converting", "frame"))
                if done:
                    result = (f"{'Sample saved' if sample else 'Saved'}: {os.path.basename(dst)}", dst, "video")
        except Exception as exc:
            err_msg = str(exc)
            print(f"[Job error] {err_msg}")
            self.after(0, lambda: messagebox.showerror("Conversion failed", f"Could not convert:\n{err_msg}"))
        finally:
            self.after(0, lambda: self._job_finished(result, tip))

    def _job_finished(self, result, tip):
        self._set_busy(False)
        self.stats_label.configure(text="")
        if result:
            self.progress_bar.set(1.0)
            self._show_result(*result, tip=tip)
        elif self.cancel_event.is_set():
            self._set_status("Cancelled.")
        else:
            self._set_status("Ready.")


def main():
    app = Any2VRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
