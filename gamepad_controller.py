"""
Virtual gamepad controller using ViGEmBus (via vgamepad).

Turns a screen-space aim point into analog right-stick deflection,
so games / apps that read controller input see the aim move.

Not a replacement for MouseController — a parallel option.
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Tuple

try:
    import vgamepad as vg
    _VG_AVAILABLE = True
    _VG_IMPORT_ERROR = None
except Exception as e:
    _VG_AVAILABLE = False
    _VG_IMPORT_ERROR = str(e)


def is_available() -> Tuple[bool, str]:
    """Return (ok, message). Use to check before starting."""
    if _VG_AVAILABLE:
        return True, "vgamepad import OK"
    return False, (
        f"vgamepad not available: {_VG_IMPORT_ERROR}. "
        f"Install with: python -m pip install vgamepad"
    )


class GamepadController:
    """
    Mirrors MouseController's interface (select_target, update, enabled),
    but drives a virtual Xbox 360 pad instead of the cursor.

    Aim mapping:
      - Compute the pixel offset between the screen center and the
        target aim point.
      - Normalize by `max_offset_px` to get a value in [-1, 1].
      - Apply a deadzone so small offsets don't jitter the stick.
      - Apply a gain so the stick saturates before the offset does.
      - Send the stick value to the virtual pad and update().
    """

    def __init__(self,
                 max_offset_px: int = 400,
                 deadzone: float = 0.08,
                 gain: float = 1.6,
                 target_class_ids: Optional[List[int]] = None,
                 click_on_detect: bool = False,
                 click_delay: float = 0.05,
                 aim_offset_x: int = 0,
                 aim_offset_y: int = 0,
                 aim_at_top: bool = False,
                 self_exclude_px: int = 0):
        self.max_offset_px = max(50, int(max_offset_px))
        self.deadzone = max(0.0, min(0.5, float(deadzone)))
        self.gain = max(0.1, float(gain))
        self.target_class_ids = target_class_ids or [0]
        self.click_on_detect = click_on_detect
        self.click_delay = click_delay
        self.aim_offset_x = aim_offset_x
        self.aim_offset_y = aim_offset_y
        self.aim_at_top = aim_at_top
        self.self_exclude_px = int(self_exclude_px or 0)

        self.enabled = True
        self._lock = threading.Lock()
        self._current_target = None
        self._last_click_time = 0.0

        self._gamepad = None
        self._init_error = None

        if _VG_AVAILABLE:
            try:
                self._gamepad = vg.VX360Gamepad()
            except Exception as e:
                self._init_error = str(e)
                self._gamepad = None

        self._center = (960, 540)

    # ------------------------------------------------------------------ #
    def set_screen_center(self, x: int, y: int):
        self._center = (int(x), int(y))

    def close(self):
        with self._lock:
            if self._gamepad is not None:
                try:
                    self._gamepad.right_joystick_float(0.0, 0.0)
                    self._gamepad.left_joystick_float(0.0, 0.0)
                    self._gamepad.update()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Target selection — mirrors MouseController.select_target
    # ------------------------------------------------------------------ #
    def select_target(self, detections, screen_offset=(0, 0)):
        if not detections:
            return None

        ccx, ccy = self._center
        lock_r = getattr(self, "lock_radius_px", 0)
        lock_r2 = lock_r * lock_r if lock_r > 0 else 0
        excl_r2 = self.self_exclude_px ** 2 if self.self_exclude_px > 0 else 0

        best_pt = None
        best_dist = float("inf")
        for d in detections:
            if self.target_class_ids and d.class_id not in self.target_class_ids:
                continue

            dcx, dcy = d.center
            sx = dcx + screen_offset[0]
            sy = dcy + screen_offset[1]
            dxs = sx - ccx
            dys = sy - ccy
            d2 = dxs * dxs + dys * dys

            if lock_r2 > 0 and d2 > lock_r2:
                continue
            if excl_r2 > 0 and d2 < excl_r2:
                continue

            ax_local, ay_local = d.center
            if self.aim_at_top:
                ay_local = d.y1 + d.height * 0.15
            ax_local += self.aim_offset_x
            ay_local += self.aim_offset_y
            ax = ax_local + screen_offset[0]
            ay = ay_local + screen_offset[1]

            dist = math.hypot(ax - ccx, ay - ccy)
            if dist < best_dist:
                best_dist = dist
                best_pt = (ax, ay)
        return best_pt

    # ------------------------------------------------------------------ #
    def update(self, detections, screen_offset=(0, 0)):
        """
        Called once per worker frame. Returns the aim point
        (screen coords) or None if no target.
        """
        if not self.enabled:
            return None

        if self._gamepad is None:
            return None

        with self._lock:
            target = self.select_target(detections, screen_offset)
            if target is None:
                try:
                    self._gamepad.right_joystick_float(0.0, 0.0)
                    self._gamepad.update()
                except Exception:
                    pass
                self._current_target = None
                return None

            self._current_target = target

            cx, cy = self._center
            dx = float(target[0] - cx)
            dy = float(target[1] - cy)

            nx = dx / self.max_offset_px
            ny = dy / self.max_offset_px

            nx *= self.gain
            ny *= self.gain
            nx = max(-1.0, min(1.0, nx))
            ny = max(-1.0, min(1.0, ny))

            if abs(nx) < self.deadzone:
                nx = 0.0
            if abs(ny) < self.deadzone:
                ny = 0.0

            try:
                self._gamepad.right_joystick_float(float(nx), float(ny))
                self._gamepad.update()
            except Exception as e:
                self._init_error = str(e)
                return target

            if self.click_on_detect:
                now = time.time()
                if now - self._last_click_time >= max(self.click_delay, 0.05):
                    try:
                        self._gamepad.right_trigger_float(1.0)
                        self._gamepad.update()
                        time.sleep(0.03)
                        self._gamepad.right_trigger_float(0.0)
                        self._gamepad.update()
                    except Exception:
                        pass
                    self._last_click_time = now

            return target

    # ------------------------------------------------------------------ #
    def get_current_target(self):
        return self._current_target