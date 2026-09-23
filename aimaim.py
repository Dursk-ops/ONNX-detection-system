"""
DurskAI — Part 1/3 (engine).
UI lives in ui.py, floating overlay in window_box.py,
virtual gamepad in gamepad_controller.py.
Run:  python aimaim.py
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

# --------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------- #
try:
    import cv2
except ImportError:
    print("Missing: opencv-python  ->  python -m pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Missing: numpy  ->  python -m pip install numpy")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Missing: onnxruntime  ->  python -m pip install onnxruntime")
    sys.exit(1)

try:
    import mss
except ImportError:
    print("Missing: mss  ->  python -m pip install mss")
    sys.exit(1)

try:
    import gamepad_controller
    GAMEPAD_MODULE = gamepad_controller
except Exception:
    GAMEPAD_MODULE = None


# --------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# Fixed capture region: a 320x320 square centered on a 1920x1080
# monitor. This is hardcoded and cannot be changed from the UI.
FIXED_REGION = (800, 380, 320, 320)


def list_onnx_models() -> list:
    if not os.path.isdir(MODELS_DIR):
        return []
    try:
        files = [f for f in os.listdir(MODELS_DIR)
                 if f.lower().endswith(".onnx")]
        files.sort()
        return [os.path.join(MODELS_DIR, f) for f in files]
    except Exception:
        return []


# ===================================================================== #
# CONFIG
# ===================================================================== #
@dataclass
class DetectionConfig:
    model_path: str = ""
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    class_names: List[str] = field(default_factory=lambda: ["target"])
    use_gpu: bool = False
    num_threads: int = 4


@dataclass
class MouseConfig:
    move_duration: float = 0.15
    smooth_movement: bool = True
    target_class_ids: List[int] = field(default_factory=lambda: [0])
    click_on_detect: bool = False
    click_delay: float = 0.05
    fov_radius: int = 0
    aim_offset_x: int = 0
    aim_offset_y: int = 0
    aim_at_top: bool = False
    enabled: bool = True
    movement_type: str = "mouse"
    gamepad_max_offset_px: int = 400
    gamepad_deadzone: float = 0.08
    gamepad_gain: float = 1.6
    lock_radius_px: int = 250
    self_exclude_px: int = 90


@dataclass
class CaptureConfig:
    monitor_index: int = 1
    capture_fps: int = 60
    region: Optional[Tuple[int, int, int, int]] = FIXED_REGION


@dataclass
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    auto_start_on_load: bool = False
    last_model: str = ""
    overlay_on: bool = False
    overlay_clear: bool = True

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "AppConfig":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            return cls()
        cfg = cls()
        if "detection" in data:
            cfg.detection = DetectionConfig(**data["detection"])
        if "mouse" in data:
            cfg.mouse = MouseConfig(**data["mouse"])
        if "capture" in data:
            # Always override region with the fixed value.
            cap_data = dict(data["capture"])
            cap_data["region"] = FIXED_REGION
            cfg.capture = CaptureConfig(**cap_data)
        if "auto_start_on_load" in data:
            cfg.auto_start_on_load = bool(data["auto_start_on_load"])
        if "last_model" in data:
            cfg.last_model = str(data["last_model"])
        if "overlay_on" in data:
            cfg.overlay_on = bool(data["overlay_on"])
        if "overlay_clear" in data:
            cfg.overlay_clear = bool(data["overlay_clear"])
        return cfg


# ===================================================================== #
# DETECTOR
# ===================================================================== #
class Detection:
    __slots__ = ("x1", "y1", "x2", "y2", "confidence", "class_id")

    def __init__(self, x1, y1, x2, y2, confidence, class_id):
        self.x1 = float(x1); self.y1 = float(y1)
        self.x2 = float(x2); self.y2 = float(y2)
        self.confidence = float(confidence)
        self.class_id = int(class_id)

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


class ONNXDetector:
    def __init__(self, model_path, input_width=640, input_height=640,
                 confidence_threshold=0.5, iou_threshold=0.45,
                 class_names=None, use_gpu=False, num_threads=4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_names = class_names or []
        self.use_gpu = use_gpu

        self._session = self._create_session(use_gpu, num_threads)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_shape = inp.shape
        self._output_names = [o.name for o in self._session.get_outputs()]

        self._nchw, model_h, model_w = self._analyse_input_shape(
            self._input_shape)

        if model_w is not None and model_h is not None:
            self.input_width = int(model_w)
            self.input_height = int(model_h)
        else:
            self.input_width = int(input_width)
            self.input_height = int(input_height)

        self._warmup()

    @staticmethod
    def _analyse_input_shape(shape):
        if len(shape) != 4:
            return True, None, None

        def _dim(v):
            if isinstance(v, int) and v > 0:
                return v
            return None

        c1 = shape[1]
        c3 = shape[3]
        if c1 == 3 and c3 != 3:
            nchw = True
        elif c3 == 3 and c1 != 3:
            nchw = False
        elif c1 == 3 and c3 == 3:
            nchw = True
        else:
            h1, w1 = _dim(shape[2]), _dim(shape[3])
            nchw = bool(h1 and w1 and h1 == w1 and h1 >= 32)

        if nchw:
            h = _dim(shape[2])
            w = _dim(shape[3])
        else:
            h = _dim(shape[1])
            w = _dim(shape[2])

        return nchw, h, w

    def _create_session(self, use_gpu, num_threads):
        providers = []
        if use_gpu:
            avail = ort.get_available_providers()
            for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                      "CoreMLExecutionProvider", "ROCMExecutionProvider"):
                if p in avail:
                    providers.append(p)
        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, int(num_threads))
        opts.log_severity_level = 3

        return ort.InferenceSession(self.model_path, sess_options=opts,
                                    providers=providers)

    def _warmup(self):
        dummy = np.zeros((1, 3, self.input_height, self.input_width),
                         dtype=np.float32)
        if not self._nchw:
            dummy = np.transpose(dummy, (0, 2, 3, 1))
        try:
            self._session.run(self._output_names, {self._input_name: dummy})
        except Exception:
            pass

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        pad_w = self.input_width - new_w
        pad_h = self.input_height - new_h
        pad_x = pad_w // 2
        pad_y = pad_h // 2
        canvas = np.full((self.input_height, self.input_width, 3), 114,
                         dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)
        if not self._nchw:
            blob = np.transpose(blob, (0, 2, 3, 1))
            blob = np.ascontiguousarray(blob)
        return blob, scale, pad_x, pad_y

    def _postprocess(self, output, scale, pad_x, pad_y, ow, oh):
        if output.ndim == 3:
            output = output[0]

        if (output.shape[1] >= 6 and output.shape[0] < output.shape[1]
                and self._looks_like_e2e(output)):
            return self._parse_e2e(output, scale, pad_x, pad_y, ow, oh)

        if output.shape[0] < output.shape[1]:
            output = output.T
        return self._parse_raw(output, scale, pad_x, pad_y, ow, oh)

    @staticmethod
    def _looks_like_e2e(arr):
        last = arr[:, -1]
        if not np.all(np.isfinite(last)):
            return False
        if np.any(last < 0) or np.any(last > 1000):
            return False
        frac = np.abs(last - np.round(last))
        return float(np.mean(frac < 1e-2)) > 0.9

    def _parse_e2e(self, arr, scale, pad_x, pad_y, ow, oh):
        dets = []
        for row in arr:
            x1, y1, x2, y2, score, cls = row[:6]
            if score < self.confidence_threshold:
                continue
            x1 = float(np.clip((x1 - pad_x) / scale, 0, ow - 1))
            y1 = float(np.clip((y1 - pad_y) / scale, 0, oh - 1))
            x2 = float(np.clip((x2 - pad_x) / scale, 0, ow - 1))
            y2 = float(np.clip((y2 - pad_y) / scale, 0, oh - 1))
            dets.append(Detection(x1, y1, x2, y2, score, int(round(cls))))
        return dets

    def _parse_raw(self, arr, scale, pad_x, pad_y, ow, oh):
        n, cols = arr.shape
        if cols < 5:
            return []

        nc = len(self.class_names) if self.class_names else None
        if nc is not None:
            has_obj = (cols == 5 + nc) or (cols != 4 + nc and cols >= 6)
        else:
            has_obj = cols >= 6

        boxes = arr[:, :4]
        if has_obj:
            scores_all = arr[:, 5:]
            obj = arr[:, 4]
        else:
            scores_all = arr[:, 4:]
            obj = np.ones(n, dtype=np.float32)

        if scores_all.shape[1] == 0:
            return []

        class_ids = np.argmax(scores_all, axis=1)
        class_scores = scores_all[np.arange(n), class_ids]
        confs = obj * class_scores

        mask = confs >= self.confidence_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2.0 - pad_x) / scale
        y1 = (cy - h / 2.0 - pad_y) / scale
        x2 = (cx + w / 2.0 - pad_x) / scale
        y2 = (cy + h / 2.0 - pad_y) / scale

        x1 = np.clip(x1, 0, ow - 1)
        y1 = np.clip(y1, 0, oh - 1)
        x2 = np.clip(x2, 0, ow - 1)
        y2 = np.clip(y2, 0, oh - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        keep = self._nms(boxes_xyxy, confs, class_ids, self.iou_threshold)
        return [Detection(boxes_xyxy[i, 0], boxes_xyxy[i, 1],
                          boxes_xyxy[i, 2], boxes_xyxy[i, 3],
                          confs[i], class_ids[i]) for i in keep]

    @staticmethod
    def _nms(boxes, scores, class_ids, iou_thr):
        if boxes.shape[0] == 0:
            return []
        keep = []
        for c in np.unique(class_ids):
            idxs = np.where(class_ids == c)[0]
            b, s = boxes[idxs], scores[idxs]
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            order = s.argsort()[::-1]
            while order.size > 0:
                i = order[0]
                keep.append(int(idxs[i]))
                if order.size == 1:
                    break
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
                remaining = np.where(iou <= iou_thr)[0]
                order = order[remaining + 1]
        return keep

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []
        oh, ow = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self._session.run(self._output_names,
                                    {self._input_name: blob})
        if not outputs:
            return []
        return self._postprocess(outputs[0], scale, pad_x, pad_y, ow, oh)

    @property
    def input_size(self):
        return self.input_width, self.input_height


def draw_detections(frame, detections, class_names=None, color=(0, 255, 0)):
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


# ===================================================================== #
# SCREEN CAPTURE
# ===================================================================== #
class ScreenCapture:
    def __init__(self, monitor_index=1, target_fps=60, region=None):
        self.monitor_index = monitor_index
        self.target_fps = max(1, int(target_fps))
        # Region is always the fixed value.
        self.region = FIXED_REGION
        self._sct = None
        self._monitor = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._fps_actual = 0.0

    def _init_mss(self):
        if self._sct is None:
            self._sct = mss.mss()
        monitors = self._sct.monitors
        idx = self.monitor_index
        if idx < 0 or idx >= len(monitors):
            idx = 1 if len(monitors) > 1 else 0
            self.monitor_index = idx
        base = monitors[idx]
        left, top, width, height = FIXED_REGION
        self._monitor = {"left": base["left"] + left,
                         "top": base["top"] + top,
                         "width": width, "height": height}

    def start(self):
        if self._running:
            return
        self._init_mss()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None

    def _loop(self):
        frame_interval = 1.0 / self.target_fps
        last = time.perf_counter()
        fps_timer = last
        fps_counter = 0
        while self._running:
            now = time.perf_counter()
            elapsed = now - last
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue
            last = time.perf_counter()
            try:
                raw = self._sct.grab(self._monitor)
                frame = np.asarray(raw, dtype=np.uint8)[:, :, :3]
                frame = np.ascontiguousarray(frame)
                with self._lock:
                    self._frame = frame
            except Exception:
                time.sleep(0.05)
                continue
            fps_counter += 1
            if now - fps_timer >= 0.5:
                self._fps_actual = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def fps(self):
        return self._fps_actual

    @property
    def monitor_bounds(self):
        if self._monitor is None:
            return (0, 0, 0, 0)
        m = self._monitor
        return (m["left"], m["top"], m["width"], m["height"])

    @property
    def is_running(self):
        return self._running


# ===================================================================== #
# MOUSE CONTROLLER
# ===================================================================== #
user32 = ctypes.windll.user32

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

SM_CXSCREEN = 0
SM_CYSCREEN = 1

try:
    user32.SetProcessDPIAware()
except Exception:
    pass

user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
user32.SendInput.restype = wintypes.UINT


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


INPUT_MOUSE = 0


def _send_mouse_input(flags, dx=0, dy=0, data=0):
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = dx
    inp.union.mi.dy = dy
    inp.union.mi.mouseData = data
    inp.union.mi.dwFlags = flags
    inp.union.mi.time = 0
    inp.union.mi.dwExtraInfo = None
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


class MouseController:
    def __init__(self, move_duration=0.15, smooth_movement=True,
                 target_class_ids=None, click_on_detect=False,
                 click_delay=0.05, fov_radius=0, aim_offset_x=0,
                 aim_offset_y=0, aim_at_top=False,
                 lock_radius_px=0, self_exclude_px=0,
                 screen_center=None):
        self.move_duration = max(0.0, move_duration)
        self.smooth_movement = smooth_movement
        self.target_class_ids = target_class_ids or [0]
        self.click_on_detect = click_on_detect
        self.click_delay = click_delay
        self.fov_radius = fov_radius
        self.aim_offset_x = aim_offset_x
        self.aim_offset_y = aim_offset_y
        self.aim_at_top = aim_at_top
        self.lock_radius_px = int(lock_radius_px or 0)
        self.self_exclude_px = int(self_exclude_px or 0)
        self.enabled = True
        self._lock = threading.Lock()
        self._current_target = None
        self._last_click_time = 0.0
        self._screen_w = user32.GetSystemMetrics(SM_CXSCREEN)
        self._screen_h = user32.GetSystemMetrics(SM_CYSCREEN)
        if screen_center is not None:
            self._center = (int(screen_center[0]), int(screen_center[1]))
        else:
            self._center = (self._screen_w // 2, self._screen_h // 2)

    def set_screen_center(self, x, y):
        self._center = (int(x), int(y))

    def get_cursor_pos(self):
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def set_cursor_pos(self, x, y):
        if self._screen_w <= 1 or self._screen_h <= 1:
            return
        nx = int(x * 65535 / (self._screen_w - 1))
        ny = int(y * 65535 / (self._screen_h - 1))
        nx = max(0, min(65535, nx))
        ny = max(0, min(65535, ny))
        _send_mouse_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, nx, ny)

    def move_relative(self, dx, dy):
        if dx == 0 and dy == 0:
            return
        _send_mouse_input(MOUSEEVENTF_MOVE, dx, dy)

    def click(self, button="left"):
        if button == "left":
            down, up = MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
        elif button == "right":
            down, up = MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
        else:
            down, up = MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP
        _send_mouse_input(down)
        time.sleep(self.click_delay)
        _send_mouse_input(up)

    def select_target(self, detections, screen_offset=(0, 0)):
        if not detections:
            return None

        cursor_x, cursor_y = self.get_cursor_pos()
        ccx, ccy = self._center
        lock_r2 = self.lock_radius_px ** 2 if self.lock_radius_px > 0 else 0
        excl_r2 = self.self_exclude_px ** 2 if self.self_exclude_px > 0 else 0

        best_pt = None
        best_dist = float("inf")
        for d in detections:
            if self.target_class_ids and d.class_id not in self.target_class_ids:
                continue

            dcx, dcy = d.center
            screen_cx = dcx + screen_offset[0]
            screen_cy = dcy + screen_offset[1]

            dxs = screen_cx - ccx
            dys = screen_cy - ccy
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

            dist = math.hypot(ax - cursor_x, ay - cursor_y)
            if self.fov_radius > 0 and dist > self.fov_radius:
                continue
            if dist < best_dist:
                best_dist = dist
                best_pt = (ax, ay)
        return best_pt

    def move_to(self, target_x, target_y, duration=None):
        if not self.smooth_movement:
            self.set_cursor_pos(int(target_x), int(target_y))
            return
        dur = self.move_duration if duration is None else duration
        cx, cy = self.get_cursor_pos()
        dx = target_x - cx
        dy = target_y - cy
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            self.set_cursor_pos(int(target_x), int(target_y))
            return
        step_size = 8.0
        steps = max(2, min(120, int(dist / step_size)))
        step_time = (dur / steps) if dur > 0 else 0.0
        for i in range(1, steps + 1):
            t = i / steps
            eased = 1 - (1 - t) ** 3
            tx = cx + dx * eased
            ty = cy + dy * eased
            ncx, ncy = self.get_cursor_pos()
            rdx = int(round(tx)) - ncx
            rdy = int(round(ty)) - ncy
            if rdx or rdy:
                self.move_relative(rdx, rdy)
            if step_time > 0:
                time.sleep(step_time)

    def update(self, detections, screen_offset=(0, 0)):
        if not self.enabled:
            return None
        with self._lock:
            target = self.select_target(detections, screen_offset)
            if target is None:
                self._current_target = None
                return None
            self._current_target = target
            self.move_to(target[0], target[1])
            if self.click_on_detect:
                now = time.time()
                if now - self._last_click_time >= max(self.click_delay, 0.05):
                    self.click("left")
                    self._last_click_time = now
            return target

    def get_current_target(self):
        return self._current_target


# ===================================================================== #
# WORKER
# ===================================================================== #
class DetectionWorker:
    def __init__(self, detector, capture, mouse, on_frame=None):
        self.detector = detector
        self.capture = capture
        self.mouse = mouse
        self.on_frame = on_frame
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_infer_ms = 0.0
        self._last_total_ms = 0.0
        self._detection_count = 0
        self._loop_fps = 0.0
        self._last_aim = None
        self.last_error = None

    def start(self):
        if self._running:
            return
        self.capture.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.capture.stop()

    def _loop(self):
        frames = 0
        fps_timer = time.perf_counter()
        target_interval = 1.0 / max(1, self.capture.target_fps)

        while self._running:
            t0 = time.perf_counter()
            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            t_infer = time.perf_counter()
            try:
                all_detections = self.detector.detect(frame)
                self.last_error = None
            except Exception as e:
                self.last_error = str(e)
                all_detections = []
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            offset = (self.capture.monitor_bounds[0],
                      self.capture.monitor_bounds[1])

            # The capture region is a small 320x320 square centered on
            # the monitor, so almost every detection is already within
            # the "center region". The self-exclusion rule still applies
            # though, since the player's own character often sits in
            # that square in third-person games.
            self_excl = int(getattr(self.mouse, "self_exclude_px", 0) or 0)
            ccx, ccy = self.mouse._center
            excl_r2 = self_excl * self_excl if self_excl > 0 else 0

            fov_dets = []
            for d in all_detections:
                dcx, dcy = d.center
                sx = dcx + offset[0]
                sy = dcy + offset[1]
                dx = sx - ccx
                dy = sy - ccy
                d2 = dx * dx + dy * dy

                if excl_r2 > 0 and d2 < excl_r2:
                    continue
                fov_dets.append(d)
            detections = fov_dets

            aim_point_local = None
            try:
                if self.mouse.enabled:
                    aim_screen = self.mouse.update(detections,
                                                   screen_offset=offset)
                    if aim_screen is not None:
                        aim_point_local = (aim_screen[0] - offset[0],
                                           aim_screen[1] - offset[1])
                else:
                    probe = self.mouse.select_target(detections,
                                                     screen_offset=offset)
                    if probe is not None:
                        aim_point_local = (probe[0] - offset[0],
                                           probe[1] - offset[1])
            except Exception as e:
                self.last_error = f"aim: {e}"

            total_ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._last_infer_ms = infer_ms
                self._last_total_ms = total_ms
                self._detection_count = len(detections)
                self._last_aim = aim_point_local

            if self.on_frame is not None:
                try:
                    stats = {"infer_ms": infer_ms,
                             "total_ms": total_ms,
                             "count": len(detections),
                             "fps": self._loop_fps,
                             "aim": aim_point_local,
                             "error": self.last_error}
                    self.on_frame(frame, detections, stats)
                except Exception:
                    pass

            frames += 1
            now = time.perf_counter()
            if now - fps_timer >= 0.5:
                self._loop_fps = frames / (now - fps_timer)
                frames = 0
                fps_timer = now

            elapsed = time.perf_counter() - t0
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    @property
    def stats(self):
        with self._lock:
            return {"detections": self._detection_count,
                    "infer_ms": self._last_infer_ms,
                    "total_ms": self._last_total_ms,
                    "fps": self._loop_fps,
                    "aim": self._last_aim,
                    "error": self.last_error,
                    "running": self._running}


# ===================================================================== #
# ENTRY POINT
# ===================================================================== #
def main():
    try:
        from ui import DetectionApp
    except ImportError as e:
        print(f"Could not import UI: {e}")
        print("Make sure ui.py, window_box.py and gamepad_controller.py "
              "are in the same folder as aimaim.py.")
        sys.exit(1)

    app = DetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()