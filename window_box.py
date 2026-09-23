"""
DurskAI — floating overlay window.

Shows the 320x320 capture region (the only area the AI can see)
at 1:1 scaling, letterboxed into the window. No zoom.

- Standard window with title bar.
- PiP (transparent) or Black background.
- Top status bar with FPS / DET / error.
- Red crosshair marks the exact center of the crop.
"""

from __future__ import annotations

import sys
import tkinter as tk

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

# --------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------- #
BG_FRAME = "#16171b"
BG_BAR = "#0f1015"
BORDER = "#2a2c36"
GREEN = "#37d67a"
YELLOW = "#ffcc44"
RED = "#e5484d"
CROSSHAIR = "#ff4d4d"
FONT_MONO = "Consolas"

_TRANSPARENT_KEY = "#010203"

_TOP_BAR_H = 30
_SIDE_PAD = 6
_BOT_PAD = 6


class OverlayWindow(ctk.CTkToplevel):
    def __init__(self, master,
                 width: int = 340,
                 height: int = 380,
                 on_close=None,
                 clear: bool = True):
        super().__init__(master)
        self._master = master
        self._on_close_cb = on_close
        self._photo = None
        self._clear = bool(clear)

        self.title("DurskAI — Detector")
        self.resizable(True, True)
        self.minsize(220, 260)
        self.configure(fg_color=BG_FRAME)

        try:
            self.update_idletasks()
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            x = (sw - width) // 2
            y = (sh - height) // 2
            self.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            self.geometry(f"{width}x{height}")

        try:
            self.attributes("-topmost", True)
        except Exception:
            pass

        self._transparency_supported = sys.platform.startswith("win")

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        # Layers (bottom → top)
        self._img_id = self.canvas.create_image(0, 0, anchor="nw")

        self._cross_h_id = self.canvas.create_line(
            0, 0, 0, 0, fill=CROSSHAIR, width=1)
        self._cross_v_id = self.canvas.create_line(
            0, 0, 0, 0, fill=CROSSHAIR, width=1)

        # Top bar
        self._bar_id = self.canvas.create_rectangle(
            0, 0, 0, 0, fill=BG_BAR, outline=BORDER, width=1)
        self._fps_id = self.canvas.create_text(
            10, _TOP_BAR_H // 2, anchor="w", text="FPS --",
            fill=GREEN, font=(FONT_MONO, 11, "bold"))
        self._det_id = self.canvas.create_text(
            95, _TOP_BAR_H // 2, anchor="w", text="DET --",
            fill=YELLOW, font=(FONT_MONO, 11, "bold"))
        self._err_id = self.canvas.create_text(
            0, 0, anchor="e", text="",
            fill=RED, font=(FONT_MONO, 9, "bold"))

        self.protocol("WM_DELETE_WINDOW", self._handle_close)

        self._apply_mode()
        self.after(60, self._bootstrap_geometry)

    # ------------------------------------------------------------------ #
    def _bootstrap_geometry(self):
        try:
            self.update_idletasks()
            if self.winfo_width() < 40:
                self.geometry("340x380")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def set_clear(self, clear: bool):
        self._clear = bool(clear)
        self._apply_mode()

    def _apply_mode(self):
        if self._clear and self._transparency_supported:
            try:
                self.configure(bg=_TRANSPARENT_KEY)
                self.wm_attributes("-transparentcolor", _TRANSPARENT_KEY)
                self.canvas.configure(bg=_TRANSPARENT_KEY)
                return
            except Exception:
                pass
        try:
            self.wm_attributes("-transparentcolor", "")
        except Exception:
            pass
        try:
            self.configure(bg=BG_FRAME)
        except Exception:
            pass
        try:
            self.canvas.configure(bg=BG_FRAME)
        except Exception:
            pass

    def _handle_close(self):
        try:
            if self._on_close_cb is not None:
                self._on_close_cb()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def update_frame(self, bgr_frame, stats):
        if bgr_frame is None:
            return
        fh, fw = bgr_frame.shape[:2]
        if fw <= 0 or fh <= 0:
            return

        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 60 or ch < 60:
            return

        # Available image area (below the top bar)
        img_x0 = _SIDE_PAD
        img_y0 = _TOP_BAR_H + _SIDE_PAD
        img_x1 = cw - _SIDE_PAD
        img_y1 = ch - _BOT_PAD
        iw = int(img_x1 - img_x0)
        ih = int(img_y1 - img_y0)
        if iw < 20 or ih < 20:
            return

        # Shrink-to-fit at 1:1. Never magnify.
        scale = min(1.0, iw / fw, ih / fh)
        tw = max(1, int(fw * scale))
        th = max(1, int(fh * scale))

        if abs(scale - 1.0) < 1e-3 and fw == tw and fh == th:
            resized = bgr_frame
        else:
            interp = (cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            resized = cv2.resize(bgr_frame, (tw, th), interpolation=interp)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        self._photo = photo
        self.canvas.itemconfigure(self._img_id, image=photo)

        # Center the image inside the available area
        ox = img_x0 + (iw - tw) // 2
        oy = img_y0 + (ih - th) // 2
        self.canvas.coords(self._img_id, ox, oy)

        # Crosshair at the exact center of the displayed image
        cxm = ox + tw // 2
        cym = oy + th // 2
        arm = 8
        self.canvas.coords(self._cross_h_id,
                           cxm - arm, cym, cxm + arm, cym)
        self.canvas.coords(self._cross_v_id,
                           cxm, cym - arm, cxm, cym + arm)

        # Top bar
        self.canvas.coords(self._bar_id, 0, 0, cw, _TOP_BAR_H)
        fps = stats.get("fps", 0.0) if stats else 0.0
        count = stats.get("count", 0) if stats else 0
        err = stats.get("error") if stats else None

        self.canvas.coords(self._fps_id, 10, _TOP_BAR_H // 2)
        self.canvas.itemconfigure(self._fps_id, text=f"FPS {fps:.0f}")

        self.canvas.coords(self._det_id, 95, _TOP_BAR_H // 2)
        self.canvas.itemconfigure(self._det_id, text=f"DET {count}")

        if err:
            self.canvas.coords(self._err_id, cw - 10, _TOP_BAR_H // 2)
            self.canvas.itemconfigure(self._err_id, text=f"{err[:24]}")
        else:
            self.canvas.itemconfigure(self._err_id, text="")

        # Raise
        self.canvas.tag_raise(self._bar_id)
        self.canvas.tag_raise(self._fps_id)
        self.canvas.tag_raise(self._det_id)
        self.canvas.tag_raise(self._err_id)
        self.canvas.tag_raise(self._cross_h_id)
        self.canvas.tag_raise(self._cross_v_id)