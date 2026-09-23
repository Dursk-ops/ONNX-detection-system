"""
DurskAI — main UI.
Polished dark layout styled after the reference screenshot.

The capture region is fixed at 800, 380, 320, 320 (see aimaim.py).
There are no zoom controls — the overlay shows the crop at 1:1.

Imports engine from aimaim.py, overlay from window_box.py,
virtual gamepad from gamepad_controller.py.
Run aimaim.py, not this file.
"""

from __future__ import annotations

import os
import time
from tkinter import filedialog, messagebox

import customtkinter as ctk

from aimaim import (
    CONFIG_PATH,
    FIXED_REGION,
    GAMEPAD_MODULE,
    MODELS_DIR,
    AppConfig,
    DetectionWorker,
    MouseController,
    ONNXDetector,
    ScreenCapture,
    list_onnx_models,
)
from window_box import OverlayWindow

# --------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------- #
BG_ROOT = "#101012"
BG_SIDEBAR = "#0b0b0d"
BG_CARD = "#17181c"
BG_CARD_HOVER = "#1c1d22"
BG_INPUT = "#1f2026"
BORDER = "#242630"
BORDER_SOFT = "#1c1e26"
ACCENT = "#5b7cff"
ACCENT_DIM = "#3a4a99"
TRACK = "#2b2d36"
TRACK_FILL = "#e9eaf0"
THUMB = "#ffffff"
TEXT = "#f1f2f5"
TEXT_DIM = "#9094a0"
TEXT_FAINT = "#5b5e68"
DIVIDER = "#1e1f26"
GREEN = "#37d67a"
GREEN_HOVER = "#2bb469"
RED = "#e5484d"
RED_HOVER = "#c93a3f"

FONT = "Segoe UI"
FONT_MONO = "Consolas"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class DetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DurskAI")
        self.geometry("1080x720")
        self.minsize(980, 640)
        self.configure(fg_color=BG_ROOT)

        self.detector = None
        self.capture = None
        self.mouse = None
        self.worker = None
        self.overlay = None
        self._overlay_clear = True
        self._last_worker_error = None
        self._active_tab = "home"

        self._build_root()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<F1>", lambda e: self._toggle())
        self.bind("<F2>", lambda e: self._on_stop())

        self._load_persisted()

    # ================================================================== #
    # Root layout
    # ================================================================== #
    def _build_root(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- SIDEBAR ---- #
        sidebar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0,
                               width=238)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        logo = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo.grid(row=0, column=0, padx=26, pady=(28, 22), sticky="ew")
        ctk.CTkLabel(logo, text="DurskAI",
                     font=(FONT, 27, "bold"),
                     text_color=TEXT, anchor="w").pack(anchor="w")
        ctk.CTkLabel(logo, text="AI assistant",
                     font=(FONT, 11),
                     text_color=TEXT_FAINT, anchor="w").pack(anchor="w",
                                                             pady=(2, 0))

        ctk.CTkFrame(sidebar, fg_color=DIVIDER, height=1).grid(
            row=1, column=0, padx=22, pady=(0, 10), sticky="ew")

        self._nav_buttons = {}
        nav_items = [
            ("home", "⌂", "Home"),
            ("settings", "▦", "AI Settings"),
            ("mouse", "◈", "Miscellaneous"),
            ("keybinds", "⌘", "Keybindings"),
            ("config", "≡", "Configs"),
        ]
        r = 2
        for key, icon, label in nav_items:
            btn = ctk.CTkButton(
                sidebar,
                text=f"   {icon}     {label}",
                anchor="w", height=42, corner_radius=8,
                fg_color="transparent", hover_color=BG_CARD,
                text_color=TEXT_DIM,
                font=(FONT, 13),
                command=lambda k=key: self._show_tab(k))
            btn.grid(row=r, column=0, padx=12, pady=1, sticky="ew")
            self._nav_buttons[key] = btn
            r += 1

        ctk.CTkFrame(sidebar, fg_color=DIVIDER, height=1).grid(
            row=r, column=0, padx=22, pady=(12, 10), sticky="ew")
        r += 1

        plugins_btn = ctk.CTkButton(
            sidebar,
            text="   ⬚     Plugins",
            anchor="w", height=42, corner_radius=8,
            fg_color="transparent", hover_color=BG_CARD,
            text_color=TEXT_DIM,
            font=(FONT, 13),
            command=lambda: self._show_tab("plugins"))
        plugins_btn.grid(row=r, column=0, padx=12, pady=1, sticky="ew")
        self._nav_buttons["plugins"] = plugins_btn
        r += 1

        sidebar.grid_rowconfigure(r, weight=1)
        r += 1

        ctk.CTkFrame(sidebar, fg_color=DIVIDER, height=1).grid(
            row=r, column=0, padx=22, pady=(8, 6), sticky="ew")
        r += 1
        ctk.CTkButton(
            sidebar,
            text="   ⚙     Settings",
            anchor="w", height=38, corner_radius=8,
            fg_color="transparent", hover_color=BG_CARD,
            text_color=TEXT_DIM,
            font=(FONT, 13),
            command=lambda: self._show_tab("config")).grid(
            row=r, column=0, padx=12, pady=1, sticky="ew")
        r += 1
        ctk.CTkButton(
            sidebar,
            text="   ⓘ     About",
            anchor="w", height=38, corner_radius=8,
            fg_color="transparent", hover_color=BG_CARD,
            text_color=TEXT_DIM,
            font=(FONT, 13),
            command=lambda: self._show_tab("about")).grid(
            row=r, column=0, padx=12, pady=(1, 22), sticky="ew")

        # ---- CONTENT ---- #
        self.content = ctk.CTkFrame(self, fg_color=BG_ROOT, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(2, weight=1)

        topbar = ctk.CTkFrame(self.content, fg_color="transparent")
        topbar.grid(row=0, column=0, padx=34, pady=(30, 0), sticky="ew")
        topbar.grid_columnconfigure(0, weight=1)

        title_block = ctk.CTkFrame(topbar, fg_color="transparent")
        title_block.grid(row=0, column=0, sticky="w")
        self.lbl_page_title = ctk.CTkLabel(
            title_block, text="Home", anchor="w",
            font=(FONT, 24, "bold"), text_color=TEXT)
        self.lbl_page_title.pack(anchor="w")
        self.lbl_page_sub = ctk.CTkLabel(
            title_block, text="Assistant overview and quick controls",
            anchor="w", font=(FONT, 11), text_color=TEXT_FAINT)
        self.lbl_page_sub.pack(anchor="w", pady=(2, 0))

        master = ctk.CTkFrame(topbar, fg_color=BG_CARD, corner_radius=12,
                              border_width=1, border_color=BORDER_SOFT)
        master.grid(row=0, column=1, sticky="e")
        inner = ctk.CTkFrame(master, fg_color="transparent")
        inner.pack(padx=16, pady=10)
        self.lbl_master = ctk.CTkLabel(
            inner, text="AI Assistant: OFF", anchor="e",
            text_color=TEXT_DIM, font=(FONT, 12))
        self.lbl_master.pack(side="left", padx=(0, 12))
        self.var_master = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            inner, text="", variable=self.var_master,
            progress_color=ACCENT, button_color=THUMB,
            fg_color=TRACK, width=44,
            command=self._on_master_toggle).pack(side="left")

        ctk.CTkFrame(self.content, fg_color=DIVIDER, height=1).grid(
            row=1, column=0, padx=34, pady=(22, 6), sticky="ew")

        self.page = ctk.CTkFrame(self.content, fg_color="transparent")
        self.page.grid(row=2, column=0, padx=34, pady=(6, 6), sticky="nsew")
        self.page.grid_columnconfigure(0, weight=1)
        self.page.grid_rowconfigure(0, weight=1)

        strip = ctk.CTkFrame(self.content, fg_color="transparent")
        strip.grid(row=3, column=0, padx=34, pady=(0, 18), sticky="ew")
        strip.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(strip, text="DurskAI  ·  v1.0",
                     anchor="w", text_color=TEXT_FAINT,
                     font=(FONT, 11)).grid(row=0, column=0, sticky="w")

        pill = ctk.CTkFrame(strip, fg_color=BG_CARD, corner_radius=14,
                            border_width=1, border_color=BORDER_SOFT)
        pill.grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(pill, text="●  VERIFIED  ·  SECURE",
                     text_color=ACCENT,
                     font=(FONT, 11, "bold")).pack(padx=14, pady=6)

        self._pages = {}
        self._build_page_home()
        self._build_page_settings()
        self._build_page_mouse()
        self._build_page_keybinds()
        self._build_page_config()
        self._build_page_plugins()
        self._build_page_about()

        self._show_tab("home")

    # ================================================================== #
    # Home
    # ================================================================== #
    def _build_page_home(self):
        frame = self._make_page("home")

        self.var_move_speed = ctk.DoubleVar(value=1.50)
        self._slider_card(frame, "⏱", "Movement speed",
                          "How fast the assistant moves to the target.",
                          self.var_move_speed, 0.0, 3.0, "{:.2f}",
                          self._on_move_speed_change, row=0)

        self.var_aim_height = ctk.DoubleVar(value=0.86)
        self._slider_card(frame, "◎", "Aim height",
                          "Vertical position on the target box to aim at.",
                          self.var_aim_height, 0.0, 1.0, "{:.2f}",
                          None, row=1)

        self.var_conf = ctk.DoubleVar(value=0.25)
        self._slider_card(frame, "◈", "AI confidence threshold",
                          "Minimum detection confidence to act on.",
                          self.var_conf, 0.05, 0.99, "{:.2f}",
                          None, row=2)

        self.var_movement_type = ctk.StringVar(value="mouse")
        self._dropdown_card(frame, "⇄", "Movement type",
                            "Mouse = real cursor via SendInput. Gamepad = ViGEmBus virtual pad.",
                            self.var_movement_type,
                            ["mouse", "gamepad"],
                            row=3)

        run_card = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12,
                                border_width=1, border_color=BORDER_SOFT)
        run_card.grid(row=4, column=0, padx=0, pady=(14, 8), sticky="ew")
        run_card.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_start = ctk.CTkButton(
            run_card, text="START", height=44, corner_radius=10,
            fg_color=GREEN, hover_color=GREEN_HOVER,
            text_color="#07200f",
            font=(FONT, 14, "bold"),
            command=self._on_start)
        self.btn_start.grid(row=0, column=0, padx=(16, 8), pady=16,
                            sticky="ew")

        self.btn_stop = ctk.CTkButton(
            run_card, text="STOP", height=44, corner_radius=10,
            fg_color=RED, hover_color=RED_HOVER,
            font=(FONT, 14, "bold"),
            state="disabled", command=self._on_stop)
        self.btn_stop.grid(row=0, column=1, padx=8, pady=16, sticky="ew")

        self.btn_overlay = ctk.CTkButton(
            run_card, text="Show overlay", height=44, corner_radius=10,
            fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
            border_width=1, border_color=BORDER,
            font=(FONT, 14, "bold"),
            command=self._toggle_overlay)
        self.btn_overlay.grid(row=0, column=2, padx=(8, 16), pady=16,
                              sticky="ew")

    # ================================================================== #
    # AI Settings
    # ================================================================== #
    def _build_page_settings(self):
        frame = self._make_page("settings")
        row = 0

        self.var_model_path = ctk.StringVar(value="")
        model_row = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12,
                                 border_width=1, border_color=BORDER_SOFT)
        model_row.grid(row=row, column=0, padx=0, pady=6, sticky="ew")
        model_row.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(model_row, text="▤", text_color=TEXT_DIM,
                     font=(FONT, 18)).grid(
            row=0, column=0, rowspan=2, padx=(20, 16), pady=18, sticky="w")
        ctk.CTkLabel(model_row, text="ONNX model", anchor="w",
                     text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, pady=(18, 0), sticky="w")
        ctk.CTkLabel(model_row, text="Pick a file from the models/ folder.",
                     anchor="w", text_color=TEXT_DIM,
                     font=(FONT, 11)).grid(
            row=1, column=1, pady=(0, 18), sticky="w")

        self.model_dropdown = ctk.CTkOptionMenu(
            model_row, variable=self.var_model_path, values=["(none)"],
            fg_color=BG_INPUT, button_color=BG_INPUT,
            button_hover_color=TRACK, dropdown_fg_color=BG_CARD,
            corner_radius=8,
            command=lambda _v: self._on_model_changed())
        self.model_dropdown.grid(row=0, column=2, rowspan=2,
                                 padx=(14, 6), pady=18, sticky="e")
        ctk.CTkButton(model_row, text="Browse", width=84, corner_radius=8,
                      fg_color=ACCENT, hover_color=ACCENT_DIM,
                      command=self._browse_model).grid(
            row=0, column=3, rowspan=2, padx=(0, 6), pady=18)
        ctk.CTkButton(model_row, text="↻", width=40, corner_radius=8,
                      fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
                      command=self._refresh_models).grid(
            row=0, column=4, rowspan=2, padx=(0, 18), pady=18)

        self.lbl_model_path = ctk.CTkLabel(
            frame, text="", anchor="w", wraplength=680,
            text_color=TEXT_FAINT, font=(FONT_MONO, 10))
        self.lbl_model_path.grid(row=row, column=0, padx=4, pady=(0, 8),
                                 sticky="ew")
        row += 1

        self.var_input_w = ctk.IntVar(value=320)
        self.var_input_h = ctk.IntVar(value=320)
        self._spin_pair_card(frame, "▣", "Input size (W / H)",
                             "Auto-filled from the model if it uses a fixed size.",
                             self.var_input_w, self.var_input_h, row=row)
        row += 1

        self.var_iou = ctk.DoubleVar(value=0.45)
        self._slider_card(frame, "◉", "NMS IoU",
                          "Overlap threshold for box merging.",
                          self.var_iou, 0.05, 0.95, "{:.2f}", None, row=row)
        row += 1

        self.var_classes = ctk.StringVar(value="target")
        self._entry_card(frame, "☰", "Class names (comma-separated)",
                         "Display names for each class index.",
                         self.var_classes, row=row)
        row += 1

        self.var_target_ids = ctk.StringVar(value="0")
        self._entry_card(frame, "⊙", "Target class IDs (comma-separated)",
                         "Which classes the assistant is allowed to aim at.",
                         self.var_target_ids, row=row)
        row += 1

        self.var_threads = ctk.IntVar(value=4)
        self._spin_card(frame, "⚙", "CPU threads",
                        "ONNX inference threads.",
                        self.var_threads, row=row)
        row += 1

        self.var_use_gpu = ctk.BooleanVar(value=False)
        self._switch_card(frame, "⚡", "Use GPU (CUDA / DirectML)",
                          "Requires onnxruntime-gpu with a supported GPU.",
                          self.var_use_gpu, row=row)

    # ================================================================== #
    # Miscellaneous
    # ================================================================== #
    def _build_page_mouse(self):
        frame = self._make_page("mouse")
        row = 0

        self.var_mouse_enabled = ctk.BooleanVar(value=True)
        self._switch_card(frame, "◉", "Enable movement",
                          "Master toggle for cursor / stick output.",
                          self.var_mouse_enabled, row=row)
        row += 1

        self.var_self_exclude = ctk.IntVar(value=0)
        self._spin_card(frame, "⊘", "Self-exclusion radius (px)",
                        "Ignore anything this close to screen center (0 = off).",
                        self.var_self_exclude, row=row)
        row += 1

        self.var_smooth = ctk.BooleanVar(value=True)
        self._switch_card(frame, "〜", "Smooth movement (mouse mode)",
                          "Interpolate the cursor path instead of snapping.",
                          self.var_smooth, row=row)
        row += 1

        self.var_move_dur = ctk.DoubleVar(value=0.15)
        self._slider_card(frame, "◔", "Move duration (s, mouse mode)",
                          "How long each move takes.",
                          self.var_move_dur, 0.0, 1.0, "{:.2f}",
                          None, row=row)
        row += 1

        self.var_aim_top = ctk.BooleanVar(value=False)
        self._switch_card(frame, "▲", "Aim at top of box",
                          "Aim near the head instead of the center.",
                          self.var_aim_top, row=row)
        row += 1

        self.var_off_x = ctk.IntVar(value=0)
        self.var_off_y = ctk.IntVar(value=0)
        self._spin_pair_card(frame, "✛", "Aim offset (X / Y)",
                             "Pixel offset applied to the aim point.",
                             self.var_off_x, self.var_off_y, row=row)
        row += 1

        self.var_click = ctk.BooleanVar(value=False)
        self._switch_card(frame, "⦿", "Click on detect",
                          "Left click (mouse) / right trigger (gamepad).",
                          self.var_click, row=row)
        row += 1

        self.var_gamepad_max_offset = ctk.IntVar(value=400)
        self._spin_card(frame, "◧", "Gamepad max offset (px)",
                        "Pixels from screen center for full stick deflection.",
                        self.var_gamepad_max_offset, row=row)
        row += 1

        self.var_gamepad_deadzone = ctk.DoubleVar(value=0.08)
        self._slider_card(frame, "○", "Gamepad deadzone",
                          "Ignore stick values below this threshold.",
                          self.var_gamepad_deadzone, 0.0, 0.4,
                          "{:.2f}", None, row=row)
        row += 1

        self.var_gamepad_gain = ctk.DoubleVar(value=1.6)
        self._slider_card(frame, "◈", "Gamepad gain",
                          "Multiplier applied to stick output.",
                          self.var_gamepad_gain, 0.2, 3.0,
                          "{:.2f}", None, row=row)

    # ================================================================== #
    # Keybindings
    # ================================================================== #
    def _build_page_keybinds(self):
        frame = self._make_page("keybinds")
        self._info_row(frame, "⌘", "Toggle assistant", "F1", row=0)
        self._info_row(frame, "⌘", "Stop", "F2", row=1)

    # ================================================================== #
    # Configs
    # ================================================================== #
    def _build_page_config(self):
        frame = self._make_page("config")
        row = 0

        self.var_monitor = ctk.IntVar(value=1)
        self._spin_card(frame, "▭", "Monitor index",
                        "0 = all monitors, 1 = primary.",
                        self.var_monitor, row=row)
        row += 1

        self.var_fps = ctk.IntVar(value=60)
        self._spin_card(frame, "▶", "Capture FPS",
                        "Higher = smoother, more CPU.",
                        self.var_fps, row=row)
        row += 1

        # Fixed-region info row (read-only)
        region_str = ", ".join(str(v) for v in FIXED_REGION)
        self._info_row(frame, "▢", "Capture region (locked)",
                       f"{region_str}   [{FIXED_REGION[2]}x{FIXED_REGION[3]}]",
                       row=row, dim=True)
        row += 1

        mode_row = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12,
                                border_width=1, border_color=BORDER_SOFT)
        mode_row.grid(row=row, column=0, padx=0, pady=6, sticky="ew")
        mode_row.grid_columnconfigure(1, weight=1)
        row += 1
        ctk.CTkLabel(mode_row, text="◐", text_color=TEXT_DIM,
                     font=(FONT, 18)).grid(
            row=0, column=0, rowspan=2, padx=(20, 16), pady=18, sticky="w")
        ctk.CTkLabel(mode_row, text="Overlay background", anchor="w",
                     text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, pady=(18, 0), sticky="w")
        ctk.CTkLabel(mode_row, text="PiP = transparent. Black = solid.",
                     anchor="w", text_color=TEXT_DIM,
                     font=(FONT, 11)).grid(
            row=1, column=1, pady=(0, 18), sticky="w")
        self.btn_overlay_mode = ctk.CTkButton(
            mode_row, text="PiP", width=110, corner_radius=8,
            fg_color=BG_INPUT, hover_color=BG_CARD_HOVER,
            border_width=1, border_color=BORDER,
            command=self._toggle_overlay_mode)
        self.btn_overlay_mode.grid(row=0, column=2, rowspan=2,
                                   padx=18, pady=18)

        io_row = ctk.CTkFrame(frame, fg_color="transparent")
        io_row.grid(row=row, column=0, padx=0, pady=(14, 8), sticky="ew")
        io_row.grid_columnconfigure((0, 1), weight=1)
        row += 1
        ctk.CTkButton(io_row, text="Save config", height=40, corner_radius=8,
                      fg_color=BG_CARD, hover_color=BG_CARD_HOVER,
                      border_width=1, border_color=BORDER,
                      command=self._on_save_config).grid(
            row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(io_row, text="Load config", height=40, corner_radius=8,
                      fg_color=BG_CARD, hover_color=BG_CARD_HOVER,
                      border_width=1, border_color=BORDER,
                      command=self._on_load_config).grid(
            row=0, column=1, padx=(6, 0), sticky="ew")

        ctk.CTkButton(frame, text="Test model on one frame", height=40,
                      corner_radius=8,
                      fg_color=BG_CARD_HOVER, hover_color=BG_INPUT,
                      border_width=1, border_color=BORDER,
                      command=self._on_test).grid(
            row=row, column=0, padx=0, pady=(6, 8), sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="LOG", anchor="w",
                     text_color=TEXT_DIM,
                     font=(FONT, 10, "bold")).grid(
            row=row, column=0, padx=4, pady=(10, 4), sticky="w")
        row += 1
        self.log_box = ctk.CTkTextbox(
            frame, height=150, fg_color=BG_CARD, corner_radius=10,
            border_width=1, border_color=BORDER_SOFT,
            text_color=TEXT_DIM, font=(FONT_MONO, 11))
        self.log_box.grid(row=row, column=0, padx=0, pady=(0, 12), sticky="ew")
        self.log_box.configure(state="disabled")

    # ================================================================== #
    # Plugins / About
    # ================================================================== #
    def _build_page_plugins(self):
        frame = self._make_page("plugins")
        self._info_row(frame, "⬚", "Custom post-processors", "—", row=0,
                       dim=True)
        self._info_row(frame, "⬚", "Custom aim rules", "—", row=1, dim=True)

    def _build_page_about(self):
        frame = self._make_page("about")
        self._info_row(frame, "ⓘ", "Name", "DurskAI", row=0)
        self._info_row(frame, "▤", "Version", "1.0", row=1)
        self._info_row(frame, "⚙", "Backend", "onnxruntime + mss + Win32",
                       row=2)
        self._info_row(frame, "▢", "Config path", CONFIG_PATH, row=3,
                       dim=True)

    # ================================================================== #
    # Card builders
    # ================================================================== #
    def _make_page(self, key):
        page = ctk.CTkScrollableFrame(
            self.page, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=BG_CARD,
            scrollbar_button_hover_color=BG_CARD_HOVER)
        page.grid_columnconfigure(0, weight=1)
        self._pages[key] = page
        return page

    def _card(self, page, row):
        card = ctk.CTkFrame(page, fg_color=BG_CARD, corner_radius=12,
                            border_width=1, border_color=BORDER_SOFT)
        card.grid(row=row, column=0, padx=0, pady=6, sticky="ew")
        return card

    def _icon_chip(self, parent, icon):
        chip = ctk.CTkFrame(parent, fg_color=BG_INPUT, corner_radius=8,
                            width=38, height=38)
        chip.grid_propagate(False)
        ctk.CTkLabel(chip, text=icon, text_color=TEXT_DIM,
                     font=(FONT, 17)).place(relx=0.5, rely=0.5,
                                             anchor="center")
        return chip

    def _slider_card(self, page, icon, label, subtitle, var, frm, to, fmt,
                     on_change, row=None):
        if row is None:
            row = getattr(page, "_next_row", 0)
        page._next_row = row + 1

        card = self._card(page, row)
        card.grid_columnconfigure(2, weight=1)

        chip = self._icon_chip(card, icon)
        chip.grid(row=0, column=0, rowspan=3, padx=(18, 14), pady=18,
                  sticky="w")

        ctk.CTkLabel(card, text=label, anchor="w", text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, columnspan=2, pady=(18, 0), sticky="w")
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, anchor="w",
                         text_color=TEXT_DIM,
                         font=(FONT, 11)).grid(
                row=1, column=1, columnspan=2, pady=(0, 4), sticky="w")

        val_lbl = ctk.CTkLabel(card, text=fmt.format(var.get()),
                               anchor="e", text_color=TEXT,
                               font=(FONT, 12, "bold"),
                               fg_color=BG_INPUT, corner_radius=6,
                               width=64, height=28)
        val_lbl.grid(row=0, column=3, padx=(0, 18), pady=(18, 0),
                     sticky="e")

        def _on_change(v):
            s = fmt.format(float(v))
            val_lbl.configure(text=s)
            if on_change is not None:
                try:
                    on_change()
                except Exception:
                    pass

        ctk.CTkSlider(card, from_=frm, to=to, variable=var,
                      progress_color=TRACK_FILL,
                      button_color=THUMB,
                      button_hover_color="#dcdce0",
                      fg_color=TRACK,
                      height=14,
                      command=_on_change).grid(
            row=2, column=1, columnspan=3, padx=(0, 18), pady=(10, 18),
            sticky="ew")

    def _entry_card(self, page, icon, label, subtitle, var, row=None):
        if row is None:
            row = getattr(page, "_next_row", 0)
        page._next_row = row + 1

        card = self._card(page, row)
        card.grid_columnconfigure(2, weight=1)

        chip = self._icon_chip(card, icon)
        chip.grid(row=0, column=0, rowspan=3, padx=(18, 14), pady=18,
                  sticky="w")

        ctk.CTkLabel(card, text=label, anchor="w", text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, columnspan=2, pady=(18, 0), sticky="w")
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, anchor="w",
                         text_color=TEXT_DIM,
                         font=(FONT, 11)).grid(
                row=1, column=1, columnspan=2, pady=(0, 4), sticky="w")

        ctk.CTkEntry(card, textvariable=var, fg_color=BG_INPUT,
                     border_color=BORDER, text_color=TEXT,
                     corner_radius=8, height=34).grid(
            row=2, column=1, columnspan=3, padx=(0, 18), pady=(6, 18),
            sticky="ew")

    def _spin_card(self, page, icon, label, subtitle, var, row=None):
        if row is None:
            row = getattr(page, "_next_row", 0)
        page._next_row = row + 1

        card = self._card(page, row)
        card.grid_columnconfigure(2, weight=1)

        chip = self._icon_chip(card, icon)
        chip.grid(row=0, column=0, rowspan=2, padx=(18, 14), pady=18,
                  sticky="w")

        ctk.CTkLabel(card, text=label, anchor="w", text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, pady=(18, 0), sticky="w")
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, anchor="w",
                         text_color=TEXT_DIM,
                         font=(FONT, 11)).grid(
                row=1, column=1, pady=(0, 18), sticky="w")

        ctk.CTkEntry(card, textvariable=var, fg_color=BG_INPUT,
                     border_color=BORDER, text_color=TEXT,
                     corner_radius=8, height=34, width=110).grid(
            row=0, column=3, rowspan=2, padx=(0, 18), pady=18, sticky="e")

    def _spin_pair_card(self, page, icon, label, subtitle, var_a, var_b,
                        row=None):
        if row is None:
            row = getattr(page, "_next_row", 0)
        page._next_row = row + 1

        card = self._card(page, row)
        card.grid_columnconfigure(2, weight=1)

        chip = self._icon_chip(card, icon)
        chip.grid(row=0, column=0, rowspan=2, padx=(18, 14), pady=18,
                  sticky="w")

        ctk.CTkLabel(card, text=label, anchor="w", text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, pady=(18, 0), sticky="w")
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, anchor="w",
                         text_color=TEXT_DIM,
                         font=(FONT, 11)).grid(
                row=1, column=1, pady=(0, 18), sticky="w")

        pair = ctk.CTkFrame(card, fg_color="transparent")
        pair.grid(row=0, column=3, rowspan=2, padx=(0, 18), pady=18)
        ctk.CTkEntry(pair, textvariable=var_a, width=90, height=34,
                     fg_color=BG_INPUT, border_color=BORDER,
                     corner_radius=8, text_color=TEXT).pack(side="left")
        ctk.CTkEntry(pair, textvariable=var_b, width=90, height=34,
                     fg_color=BG_INPUT, border_color=BORDER,
                     corner_radius=8, text_color=TEXT).pack(side="left",
                                                             padx=(6, 0))

    def _switch_card(self, page, icon, label, subtitle, var, row=None):
        if row is None:
            row = getattr(page, "_next_row", 0)
        page._next_row = row + 1

        card = self._card(page, row)
        card.grid_columnconfigure(2, weight=1)

        chip = self._icon_chip(card, icon)
        chip.grid(row=0, column=0, rowspan=2, padx=(18, 14), pady=18,
                  sticky="w")

        ctk.CTkLabel(card, text=label, anchor="w", text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, pady=(18, 0), sticky="w")
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, anchor="w",
                         text_color=TEXT_DIM,
                         font=(FONT, 11)).grid(
                row=1, column=1, pady=(0, 18), sticky="w")

        ctk.CTkSwitch(card, text="", variable=var,
                      progress_color=ACCENT,
                      button_color=THUMB,
                      fg_color=TRACK).grid(
            row=0, column=3, rowspan=2, padx=(0, 18), pady=18, sticky="e")

    def _dropdown_card(self, page, icon, label, subtitle, var, values,
                       row=None):
        if row is None:
            row = getattr(page, "_next_row", 0)
        page._next_row = row + 1

        card = self._card(page, row)
        card.grid_columnconfigure(2, weight=1)

        chip = self._icon_chip(card, icon)
        chip.grid(row=0, column=0, rowspan=2, padx=(18, 14), pady=18,
                  sticky="w")

        ctk.CTkLabel(card, text=label, anchor="w", text_color=TEXT,
                     font=(FONT, 13, "bold")).grid(
            row=0, column=1, pady=(18, 0), sticky="w")
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, anchor="w",
                         text_color=TEXT_DIM,
                         font=(FONT, 11)).grid(
                row=1, column=1, pady=(0, 18), sticky="w")

        ctk.CTkOptionMenu(card, variable=var, values=values,
                          width=160, fg_color=BG_INPUT,
                          button_color=BG_INPUT,
                          button_hover_color=TRACK,
                          corner_radius=8,
                          dropdown_fg_color=BG_CARD).grid(
            row=0, column=3, rowspan=2, padx=(0, 18), pady=18, sticky="e")

    def _info_row(self, page, icon, label, value, row, dim=False):
        card = self._card(page, row)
        card.grid_columnconfigure(2, weight=1)

        chip = self._icon_chip(card, icon)
        chip.grid(row=0, column=0, padx=(18, 14), pady=16, sticky="w")

        ctk.CTkLabel(card, text=label, anchor="w", text_color=TEXT,
                     font=(FONT, 13)).grid(
            row=0, column=1, pady=16, sticky="w")
        ctk.CTkLabel(card, text=value, anchor="e",
                     text_color=TEXT_DIM if dim else TEXT,
                     font=(FONT, 12, "bold")).grid(
            row=0, column=3, padx=(0, 18), pady=16, sticky="e")

    # ================================================================== #
    # Navigation
    # ================================================================== #
    def _show_tab(self, key):
        if key not in self._pages:
            return
        for k, page in self._pages.items():
            if k == key:
                page.grid(row=0, column=0, sticky="nsew")
            else:
                page.grid_forget()

        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.configure(fg_color=BG_CARD, text_color=TEXT)
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_DIM)

        titles = {
            "home": ("Home", "Assistant overview and quick controls"),
            "settings": ("AI Settings", "Model, thresholds, class selection"),
            "mouse": ("Miscellaneous", "Mouse and gamepad tuning"),
            "keybinds": ("Keybindings", "Global hotkeys"),
            "config": ("Configs", "Capture, overlay and file I/O"),
            "plugins": ("Plugins", "Reserved for future extensions"),
            "about": ("About", "Build info"),
        }
        t = titles.get(key, (key.title(), ""))
        try:
            self.lbl_page_title.configure(text=t[0])
            self.lbl_page_sub.configure(text=t[1])
        except Exception:
            pass
        self._active_tab = key

    # ================================================================== #
    # Master switch
    # ================================================================== #
    def _on_master_toggle(self):
        if self.var_master.get():
            self._on_start()
        else:
            self._on_stop()

    def _set_master_display(self, on):
        self.var_master.set(bool(on))
        try:
            self.lbl_master.configure(
                text="AI Assistant: ON" if on else "AI Assistant: OFF",
                text_color=GREEN if on else TEXT_DIM)
        except Exception:
            pass

    # ================================================================== #
    # Parsers / config
    # ================================================================== #
    def _parse_classes(self, s):
        return [c.strip() for c in s.split(",") if c.strip()]

    def _parse_ids(self, s):
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok.lstrip("-").isdigit():
                out.append(int(tok))
        return out or [0]

    def _gather_config(self):
        cfg = AppConfig()
        cfg.detection.model_path = self.var_model_path.get().strip()
        cfg.detection.input_width = int(self.var_input_w.get())
        cfg.detection.input_height = int(self.var_input_h.get())
        cfg.detection.confidence_threshold = float(self.var_conf.get())
        cfg.detection.iou_threshold = float(self.var_iou.get())
        cfg.detection.class_names = self._parse_classes(self.var_classes.get())
        cfg.detection.num_threads = int(self.var_threads.get())
        cfg.detection.use_gpu = bool(self.var_use_gpu.get())

        cfg.capture.monitor_index = int(self.var_monitor.get())
        cfg.capture.capture_fps = int(self.var_fps.get())
        cfg.capture.region = FIXED_REGION  # always

        cfg.mouse.enabled = bool(self.var_mouse_enabled.get())
        cfg.mouse.smooth_movement = bool(self.var_smooth.get())
        cfg.mouse.move_duration = float(self.var_move_dur.get())
        cfg.mouse.target_class_ids = self._parse_ids(self.var_target_ids.get())
        cfg.mouse.aim_at_top = bool(self.var_aim_top.get())
        cfg.mouse.aim_offset_x = int(self.var_off_x.get())
        cfg.mouse.aim_offset_y = int(self.var_off_y.get())
        cfg.mouse.click_on_detect = bool(self.var_click.get())
        cfg.mouse.movement_type = self.var_movement_type.get().strip() or "mouse"
        cfg.mouse.gamepad_max_offset_px = int(self.var_gamepad_max_offset.get())
        cfg.mouse.gamepad_deadzone = float(self.var_gamepad_deadzone.get())
        cfg.mouse.gamepad_gain = float(self.var_gamepad_gain.get())
        cfg.mouse.self_exclude_px = int(self.var_self_exclude.get())

        cfg.auto_start_on_load = False
        cfg.last_model = self.var_model_path.get().strip()
        cfg.overlay_on = self.overlay is not None
        cfg.overlay_clear = bool(self._overlay_clear)
        return cfg

    def _resolve_model_path(self, raw):
        raw = (raw or "").strip()
        if not raw:
            return ""
        if os.path.isabs(raw):
            return raw if os.path.exists(raw) else ""
        if os.path.exists(raw):
            return os.path.abspath(raw)
        candidate = os.path.join(MODELS_DIR, raw)
        if os.path.exists(candidate):
            return candidate
        return ""

    # ================================================================== #
    # Model picker
    # ================================================================== #
    def _browse_model(self):
        initial = MODELS_DIR if os.path.isdir(MODELS_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title="Select ONNX model",
            initialdir=initial,
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            same = os.path.samefile(os.path.dirname(os.path.abspath(path)),
                                    MODELS_DIR)
        except Exception:
            same = False
        self.var_model_path.set(
            os.path.basename(path) if same else path)
        self._update_model_path_label()
        self._save_persisted()

    def _refresh_models(self):
        files = list_onnx_models()
        names = [os.path.basename(p) for p in files]
        if not names:
            names = ["(no .onnx files in models/)"]
        try:
            self.model_dropdown.configure(values=names)
        except Exception:
            pass

        current = self.var_model_path.get().strip()
        valid = names and names[0] != "(no .onnx files in models/)"
        if valid and (not current or current not in names
                      and not os.path.isabs(current)):
            self.var_model_path.set(names[0])
        self._update_model_path_label()

    def _update_model_path_label(self):
        v = self.var_model_path.get().strip()
        if not v:
            self.lbl_model_path.configure(text="(no model selected)")
            return
        if os.path.isabs(v):
            exists = "✓" if os.path.exists(v) else "✗"
            self.lbl_model_path.configure(text=f"{exists} {v}")
        else:
            full = os.path.join(MODELS_DIR, v)
            exists = "✓" if os.path.exists(full) else "✗"
            self.lbl_model_path.configure(text=f"{exists} {full}")

    def _on_model_changed(self):
        self._update_model_path_label()
        self._save_persisted()

    # ================================================================== #
    # Persistence
    # ================================================================== #
    def _load_persisted(self):
        cfg = AppConfig.load(CONFIG_PATH)

        self.var_input_w.set(cfg.detection.input_width)
        self.var_input_h.set(cfg.detection.input_height)
        self.var_conf.set(cfg.detection.confidence_threshold)
        self.var_iou.set(cfg.detection.iou_threshold)
        if cfg.detection.class_names:
            self.var_classes.set(",".join(cfg.detection.class_names))
        self.var_threads.set(cfg.detection.num_threads)
        self.var_use_gpu.set(cfg.detection.use_gpu)

        self.var_monitor.set(cfg.capture.monitor_index)
        self.var_fps.set(cfg.capture.capture_fps)

        self.var_mouse_enabled.set(cfg.mouse.enabled)
        self.var_smooth.set(cfg.mouse.smooth_movement)
        self.var_move_dur.set(cfg.mouse.move_duration)
        if cfg.mouse.target_class_ids:
            self.var_target_ids.set(
                ",".join(str(i) for i in cfg.mouse.target_class_ids))
        self.var_aim_top.set(cfg.mouse.aim_at_top)
        self.var_off_x.set(cfg.mouse.aim_offset_x)
        self.var_off_y.set(cfg.mouse.aim_offset_y)
        self.var_click.set(cfg.mouse.click_on_detect)
        self.var_movement_type.set(
            getattr(cfg.mouse, "movement_type", "mouse") or "mouse")
        self.var_gamepad_max_offset.set(
            int(getattr(cfg.mouse, "gamepad_max_offset_px", 400)))
        self.var_gamepad_deadzone.set(
            float(getattr(cfg.mouse, "gamepad_deadzone", 0.08)))
        self.var_gamepad_gain.set(
            float(getattr(cfg.mouse, "gamepad_gain", 1.6)))
        self.var_self_exclude.set(
            int(getattr(cfg.mouse, "self_exclude_px", 0)))

        self._overlay_clear = bool(getattr(cfg, "overlay_clear", True))
        try:
            self.btn_overlay_mode.configure(
                text="PiP" if self._overlay_clear else "Black")
        except Exception:
            pass

        for candidate in (cfg.last_model, cfg.detection.model_path):
            if not candidate:
                continue
            if self._resolve_model_path(candidate):
                self.var_model_path.set(candidate)
                break
        self._update_model_path_label()

    def _save_persisted(self):
        try:
            cfg = self._gather_config()
            cfg.save(CONFIG_PATH)
        except Exception as e:
            self._log(f"Config save failed: {e}")

    # ================================================================== #
    # Logging
    # ================================================================== #
    def _log(self, msg):
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:
            pass

    # ================================================================== #
    # Detector
    # ================================================================== #
    def _build_detector(self):
        cfg = self._gather_config()
        path = self._resolve_model_path(cfg.detection.model_path)
        if not path:
            messagebox.showerror(
                "Model missing",
                f"Model not found.\n\nPut .onnx files in:\n{MODELS_DIR}")
            return None
        try:
            det = ONNXDetector(
                model_path=path,
                input_width=cfg.detection.input_width,
                input_height=cfg.detection.input_height,
                confidence_threshold=cfg.detection.confidence_threshold,
                iou_threshold=cfg.detection.iou_threshold,
                class_names=cfg.detection.class_names,
                use_gpu=cfg.detection.use_gpu,
                num_threads=cfg.detection.num_threads,
            )
            self.var_input_w.set(det.input_width)
            self.var_input_h.set(det.input_height)
            self._log(f"Model input: {det.input_width}x{det.input_height} "
                      f"({'NCHW' if det._nchw else 'NHWC'})")
            return det
        except Exception as e:
            messagebox.showerror("Model error", f"Failed to load model:\n{e}")
            return None

    # ================================================================== #
    # Actions
    # ================================================================== #
    def _on_test(self):
        det = self._build_detector()
        if det is None:
            return
        self._log(f"Model OK: {os.path.basename(det.model_path)} "
                  f"input={det.input_size}")
        cap = ScreenCapture(monitor_index=int(self.var_monitor.get()),
                            target_fps=5)
        cap.start()
        time.sleep(0.5)
        frame = cap.get_frame()
        cap.stop()
        if frame is None:
            self._log("Capture returned no frame.")
            return
        t0 = time.perf_counter()
        try:
            dets = det.detect(frame)
            self._log(f"Test: {len(dets)} detections in "
                      f"{(time.perf_counter() - t0) * 1000.0:.1f} ms")
        except Exception as e:
            self._log(f"Test failed: {e}")

    def _on_start(self):
        if self.worker is not None:
            return
        det = self._build_detector()
        if det is None:
            self._set_master_display(False)
            return

        cfg = self._gather_config()
        try:
            self.capture = ScreenCapture(
                monitor_index=cfg.capture.monitor_index,
                target_fps=cfg.capture.capture_fps,
            )

            if (cfg.mouse.movement_type == "gamepad"
                    and GAMEPAD_MODULE is not None):
                ok, msg = GAMEPAD_MODULE.is_available()
                if not ok:
                    messagebox.showerror("Gamepad unavailable", msg)
                    return
                self.mouse = GAMEPAD_MODULE.GamepadController(
                    max_offset_px=cfg.mouse.gamepad_max_offset_px,
                    deadzone=cfg.mouse.gamepad_deadzone,
                    gain=cfg.mouse.gamepad_gain,
                    target_class_ids=cfg.mouse.target_class_ids,
                    click_on_detect=cfg.mouse.click_on_detect,
                    click_delay=cfg.mouse.click_delay,
                    aim_offset_x=cfg.mouse.aim_offset_x,
                    aim_offset_y=cfg.mouse.aim_offset_y,
                    aim_at_top=cfg.mouse.aim_at_top,
                    self_exclude_px=cfg.mouse.self_exclude_px,
                )
                try:
                    sw = self.winfo_screenwidth()
                    sh = self.winfo_screenheight()
                    self.mouse.set_screen_center(sw // 2, sh // 2)
                except Exception:
                    pass
                self._log("Movement: virtual gamepad (ViGEmBus)")
            else:
                self.mouse = MouseController(
                    move_duration=cfg.mouse.move_duration,
                    smooth_movement=cfg.mouse.smooth_movement,
                    target_class_ids=cfg.mouse.target_class_ids,
                    click_on_detect=cfg.mouse.click_on_detect,
                    click_delay=cfg.mouse.click_delay,
                    aim_offset_x=cfg.mouse.aim_offset_x,
                    aim_offset_y=cfg.mouse.aim_offset_y,
                    aim_at_top=cfg.mouse.aim_at_top,
                    self_exclude_px=cfg.mouse.self_exclude_px,
                    screen_center=(self.winfo_screenwidth() // 2,
                                   self.winfo_screenheight() // 2),
                )
                self._log("Movement: mouse (SendInput)")

            self.mouse.enabled = cfg.mouse.enabled
            self.detector = det

            self.worker = DetectionWorker(det, self.capture, self.mouse,
                                          on_frame=self._on_frame)
            self.worker.start()

            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self._set_master_display(True)
            self._log("Started.")
            self._save_persisted()
        except Exception as e:
            messagebox.showerror("Start failed", str(e))
            self._on_stop()

    def _on_stop(self):
        if self.mouse is not None and hasattr(self.mouse, "close"):
            try:
                self.mouse.close()
            except Exception:
                pass

        if self.worker is not None:
            try:
                self.worker.stop()
            except Exception:
                pass
            self.worker = None
        self.capture = None
        self.mouse = None
        try:
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass
        self._set_master_display(False)
        self._save_persisted()
        self._log("Stopped.")

    def _toggle(self):
        if self.worker is None:
            self._on_start()
        else:
            self._on_stop()

    # ================================================================== #
    # Overlay
    # ================================================================== #
    def _toggle_overlay(self):
        if self.overlay is not None and self.overlay.winfo_exists():
            self._close_overlay()
        else:
            self._open_overlay()

    def _open_overlay(self):
        if self.overlay is not None and self.overlay.winfo_exists():
            return
        self.overlay = OverlayWindow(
            self,
            on_close=self._on_overlay_closed,
            clear=self._overlay_clear)
        try:
            self.btn_overlay.configure(text="Hide overlay")
        except Exception:
            pass
        self._save_persisted()

    def _close_overlay(self):
        if self.overlay is not None:
            try:
                self.overlay.destroy()
            except Exception:
                pass
            self.overlay = None
        try:
            self.btn_overlay.configure(text="Show overlay")
        except Exception:
            pass
        self._save_persisted()

    def _on_overlay_closed(self):
        self.overlay = None
        try:
            self.btn_overlay.configure(text="Show overlay")
        except Exception:
            pass
        self._save_persisted()

    def _toggle_overlay_mode(self):
        self._overlay_clear = not self._overlay_clear
        try:
            self.btn_overlay_mode.configure(
                text="PiP" if self._overlay_clear else "Black")
        except Exception:
            pass
        if self.overlay is not None and self.overlay.winfo_exists():
            try:
                self.overlay.set_clear(self._overlay_clear)
            except Exception:
                pass
        self._save_persisted()

    def _on_move_speed_change(self):
        try:
            speed = float(self.var_move_speed.get())
            dur = max(0.0, 1.0 - (speed / 3.0))
            self.var_move_dur.set(dur)
        except Exception:
            pass

    # ================================================================== #
    # Worker callback
    # ================================================================== #
    def _on_frame(self, frame, detections, stats):
        try:
            err = stats.get("error") if stats else None
            if err and err != self._last_worker_error:
                self._last_worker_error = err
                self.after(0, self._log, f"Detection error: {err}")
            elif not err:
                self._last_worker_error = None

            if self.overlay is not None and self.overlay.winfo_exists():
                vis = _draw_detections_local(
                    frame, detections,
                    class_names=self._parse_classes(self.var_classes.get()))
                self.after(0, self.overlay.update_frame, vis, stats)
        except Exception as e:
            print("UI _on_frame error:", repr(e))

    # ================================================================== #
    # Config I/O
    # ================================================================== #
    def _on_save_config(self):
        try:
            self._save_persisted()
            self._log(f"Saved {CONFIG_PATH}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _on_load_config(self):
        try:
            cfg = AppConfig.load(CONFIG_PATH)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.var_input_w.set(cfg.detection.input_width)
        self.var_input_h.set(cfg.detection.input_height)
        self.var_conf.set(cfg.detection.confidence_threshold)
        self.var_iou.set(cfg.detection.iou_threshold)
        self.var_classes.set(",".join(cfg.detection.class_names))
        self.var_threads.set(cfg.detection.num_threads)
        self.var_use_gpu.set(cfg.detection.use_gpu)

        self.var_monitor.set(cfg.capture.monitor_index)
        self.var_fps.set(cfg.capture.capture_fps)

        self.var_mouse_enabled.set(cfg.mouse.enabled)
        self.var_smooth.set(cfg.mouse.smooth_movement)
        self.var_move_dur.set(cfg.mouse.move_duration)
        self.var_target_ids.set(
            ",".join(str(i) for i in cfg.mouse.target_class_ids))
        self.var_aim_top.set(cfg.mouse.aim_at_top)
        self.var_off_x.set(cfg.mouse.aim_offset_x)
        self.var_off_y.set(cfg.mouse.aim_offset_y)
        self.var_click.set(cfg.mouse.click_on_detect)
        self.var_movement_type.set(
            getattr(cfg.mouse, "movement_type", "mouse") or "mouse")
        self.var_gamepad_max_offset.set(
            int(getattr(cfg.mouse, "gamepad_max_offset_px", 400)))
        self.var_gamepad_deadzone.set(
            float(getattr(cfg.mouse, "gamepad_deadzone", 0.08)))
        self.var_gamepad_gain.set(
            float(getattr(cfg.mouse, "gamepad_gain", 1.6)))
        self.var_self_exclude.set(
            int(getattr(cfg.mouse, "self_exclude_px", 0)))

        self._overlay_clear = bool(getattr(cfg, "overlay_clear", True))
        try:
            self.btn_overlay_mode.configure(
                text="PiP" if self._overlay_clear else "Black")
        except Exception:
            pass

        self.var_model_path.set(cfg.last_model or cfg.detection.model_path)
        self._update_model_path_label()
        self._log(f"Loaded {CONFIG_PATH}")

    def _on_close(self):
        try:
            self._save_persisted()
            self._on_stop()
        finally:
            self.destroy()


# ------------------------------------------------------------------ #
# Draw helper for the overlay image
# ------------------------------------------------------------------ #
def _draw_detections_local(frame, detections, class_names=None,
                           color=(0, 255, 0)):
    import cv2
    for d in detections:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (class_names[d.class_id]
                 if class_names and 0 <= d.class_id < len(class_names)
                 else f"cls{d.class_id}")
        text = f"{label} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cx, cy = d.center
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return frame