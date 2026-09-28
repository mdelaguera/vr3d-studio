"""Modal windows: first-run welcome and 'how to fix' for setup issues."""

import customtkinter as ctk

from gui import theme as t

STEPS = [
    ("1", "Drop in a photo, video, or folder", "Drag it onto the window, or press Ctrl+O."),
    ("2", "Pick where you'll watch it", "VR180 for Quest / Pico headsets, Full SBS for 3D TVs."),
    ("3", "Preview, then convert", "Wiggle 3D shows the depth on a normal screen. Press Ctrl+Enter to convert."),
]


class _Modal(ctk.CTkToplevel):
    def __init__(self, parent, title, width=520):
        super().__init__(parent)
        self.title(title)
        self.configure(fg_color=t.SIDEBAR)
        self.resizable(False, False)
        self.transient(parent)
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=24, pady=20)
        self._width = width
        self.bind("<Escape>", lambda e: self.destroy())

    def show(self):
        """Center on the parent and grab focus."""
        self.update_idletasks()
        p = self.master
        x = p.winfo_rootx() + (p.winfo_width() - self._width) // 2
        y = p.winfo_rooty() + max(40, (p.winfo_height() - self.winfo_reqheight()) // 3)
        self.geometry(f"{self._width}x{self.winfo_reqheight()}+{max(0, x)}+{max(0, y)}")
        self.after(50, self.grab_set)  # Windows needs the window mapped before grabbing
        self.focus()


class WelcomeDialog(_Modal):
    """Shown once on first launch."""

    def __init__(self, parent, report, on_fix):
        super().__init__(parent, "Welcome to Any2VR")
        b = self.body
        ctk.CTkLabel(b, text="Welcome to Any2VR", font=t.font(22, True), text_color=t.TEXT).pack(anchor="w")
        ctk.CTkLabel(b, text="Turn any photo or video into 3D and VR. Everything runs privately on this computer.",
                     font=t.font(13), text_color=t.MUTED, wraplength=470, justify="left").pack(anchor="w", pady=(2, 16))

        for num, head, sub in STEPS:
            row = ctk.CTkFrame(b, fg_color=t.CARD, corner_radius=12)
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=num, width=32, height=32, corner_radius=16, fg_color=t.ACCENT,
                         text_color="white", font=t.font(14, True)).pack(side="left", padx=12, pady=10)
            txt = ctk.CTkFrame(row, fg_color="transparent")
            txt.pack(side="left", fill="x", expand=True, pady=8)
            ctk.CTkLabel(txt, text=head, font=t.font(13, True), text_color=t.TEXT).pack(anchor="w")
            ctk.CTkLabel(txt, text=sub, font=t.font(12), text_color=t.MUTED, wraplength=400, justify="left").pack(anchor="w")

        # Hardware summary
        hw = ctk.CTkFrame(b, fg_color="transparent")
        hw.pack(fill="x", pady=(16, 0))
        ok = report.has_gpu and not report.issues
        ctk.CTkLabel(hw, text=("✓ " if ok else "• ") + f"AI runs on: {report.device_label}", font=t.font(12, True),
                     text_color=t.GO if report.has_gpu else t.WARN).pack(anchor="w")
        if report.encoder:
            ctk.CTkLabel(hw, text=f"✓ Video encoding: {report.encoder}", font=t.font(12),
                         text_color=t.GO if report.encoder != "CPU" else t.MUTED).pack(anchor="w")
        for issue in report.issues:
            line = ctk.CTkFrame(hw, fg_color="transparent")
            line.pack(fill="x", pady=(4, 0))
            ctk.CTkLabel(line, text=f"• {issue.title}", font=t.font(12), text_color=t.WARN).pack(side="left")
            if issue.fix_command:
                t.button(line, "How to fix", lambda i=issue: on_fix(i), height=24, width=90).pack(side="right")

        t.button(b, "Get started", self.destroy, kind="primary", height=40, font=t.font(14, True)).pack(fill="x", pady=(20, 0))
        self.show()


class FixDialog(_Modal):
    """Explains a setup issue and offers a copyable command."""

    def __init__(self, parent, issue):
        super().__init__(parent, issue.title, width=560)
        b = self.body
        ctk.CTkLabel(b, text=issue.title, font=t.font(18, True), text_color=t.TEXT).pack(anchor="w")
        ctk.CTkLabel(b, text=issue.detail, font=t.font(13), text_color=t.MUTED, wraplength=510,
                     justify="left").pack(anchor="w", pady=(6, 12))
        if issue.fix_command:
            ctk.CTkLabel(b, text="Run this in a terminal (PowerShell):", font=t.font(12), text_color=t.TEXT).pack(anchor="w")
            box = ctk.CTkTextbox(b, height=56, font=ctk.CTkFont(family="Consolas", size=12), fg_color=t.CARD, wrap="word")
            box.insert("1.0", issue.fix_command)
            box.configure(state="disabled")
            box.pack(fill="x", pady=(4, 12))
            row = ctk.CTkFrame(b, fg_color="transparent")
            row.pack(fill="x")
            self.copy_btn = t.button(row, "Copy command", lambda: self._copy(issue.fix_command), kind="primary")
            self.copy_btn.pack(side="left")
            t.button(row, "Close", self.destroy).pack(side="right")
        else:
            t.button(b, "OK", self.destroy, kind="primary").pack(anchor="e")
        self.show()

    def _copy(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.copy_btn.configure(text="Copied ✓")
