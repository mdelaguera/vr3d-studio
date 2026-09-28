"""Palette and small widget builders shared by every window. (light, dark) tuples follow the theme."""

import customtkinter as ctk

ACCENT = ("#5B5BD6", "#7C7CF0")
ACCENT_HOVER = ("#4A4AC4", "#6868DE")
GO = ("#1F9D6B", "#23B47A")
GO_HOVER = ("#1A8A5D", "#1E9C6A")
DANGER = ("#D64545", "#E05A5A")
DANGER_HOVER = ("#BF3B3B", "#C94C4C")
WARN = ("#B7791F", "#E3A33B")
SUBTLE = ("#E4E6EE", "#2A2D3A")
SUBTLE_HOVER = ("#D5D8E3", "#343849")
CARD = ("#F4F5F9", "#1D1F29")
SIDEBAR = ("#FFFFFF", "#15161E")
MAIN_BG = ("#EBEDF3", "#0F1016")
MUTED = ("#6B7080", "#8B90A3")
TEXT = ("#1B1D26", "#E8E9F0")
PREVIEW_BG = "#0B0C10"


def font(size=12, bold=False):
    return ctk.CTkFont(size=size, weight="bold" if bold else "normal")


def card(parent, title):
    frame = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12)
    frame.pack(fill="x", padx=12, pady=(0, 10))
    ctk.CTkLabel(frame, text=title.upper(), font=font(11, True), text_color=MUTED).pack(padx=12, pady=(10, 4), anchor="w")
    return frame


def field_label(parent, text):
    ctk.CTkLabel(parent, text=text, font=font(12), text_color=TEXT).pack(padx=12, anchor="w")


def hint(parent, text="", color=MUTED):
    """Small wrapped helper text under a control."""
    lbl = ctk.CTkLabel(parent, text=text, font=font(11), text_color=color, wraplength=270, justify="left")
    lbl.pack(padx=12, pady=(0, 8), anchor="w")
    return lbl


def option_menu(parent, values, variable, command):
    menu = ctk.CTkOptionMenu(
        parent, values=values, variable=variable, command=command, height=30,
        fg_color=SUBTLE, button_color=SUBTLE_HOVER, button_hover_color=ACCENT_HOVER,
        text_color=TEXT, dynamic_resizing=False, corner_radius=8,
    )
    menu.pack(fill="x", padx=12, pady=(2, 4))
    return menu


def slider_row(parent, title, from_, to, steps, value, command):
    """Title on the left, live value on the right, slider underneath. Returns (row, value_label, slider)."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=12, pady=(2, 8))
    head = ctk.CTkFrame(row, fg_color="transparent")
    head.pack(fill="x")
    ctk.CTkLabel(head, text=title, font=font(12), text_color=TEXT).pack(side="left")
    value_lbl = ctk.CTkLabel(head, text="", font=font(12, True), text_color=ACCENT)
    value_lbl.pack(side="right")
    slider = ctk.CTkSlider(row, from_=from_, to=to, number_of_steps=steps, command=command,
                           button_color=ACCENT, button_hover_color=ACCENT_HOVER, progress_color=ACCENT)
    slider.set(value)
    slider.pack(fill="x", pady=(2, 0))
    return row, value_lbl, slider


def switch(parent, text, on, command=None):
    sw = ctk.CTkSwitch(parent, text=text, font=font(12), text_color=TEXT, progress_color=ACCENT, command=command)
    if on:
        sw.select()
    sw.pack(padx=12, pady=3, anchor="w")
    return sw


def button(parent, text, command, kind="subtle", height=32, **kw):
    """kind: primary | subtle | go | danger."""
    colors = {
        "primary": dict(fg_color=ACCENT, hover_color=ACCENT_HOVER),
        "subtle": dict(fg_color=SUBTLE, hover_color=SUBTLE_HOVER, text_color=TEXT),
        "go": dict(fg_color=GO, hover_color=GO_HOVER),
        "danger": dict(fg_color=DANGER, hover_color=DANGER_HOVER),
    }[kind]
    return ctk.CTkButton(parent, text=text, command=command, height=height,
                         corner_radius=10 if height >= 40 else 8, **colors, **kw)
