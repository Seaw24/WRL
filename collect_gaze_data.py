"""Gaze data collection with fixed ROI cropping + Pupil Neon ground truth.

Research-stage: UI calibration runs when no saved calibration is found.

Two operating modes:

  FITTING MODE (HEADLESS_MODE = False):
    If no calibration JSON exists, opens an interactive calibration window.
    Use this on your laptop, or over SSH + X forwarding from a laptop to the
    Pi. This is the mode you want during research.

  RUN MODE (HEADLESS_MODE = True):
    Requires an existing calibration JSON. No UI is ever shown. Reserved for
    the eventual Pi deployment - not used during research.

    For real deployment later, the proper fix is session-start auto-
    calibration (e.g. Neon-assisted, or a physical button + one-shot
    detector). That's a TODO, not a today.

Calibration controls:
    Left click        : set eye center
    W/A/S/D           : nudge center by 2px
    + / -             : grow / shrink the crop box
    ENTER             : confirm and save
    ESC               : cancel

During calibration, roll your eyes through the full gaze range (up, down,
left, right, diagonals). The pupil must stay inside the green box at every
extreme. If it leaves, press + to grow the box.
"""

import contextlib
import json
import os
import socket
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

try:
    from pupil_labs.realtime_api.simple import Device as NeonDevice
    from pupil_labs.realtime_api.simple import discover_one_device
except ImportError as exc:
    NeonDevice = None
    discover_one_device = None
    PUPIL_LABS_IMPORT_ERROR = exc
else:
    PUPIL_LABS_IMPORT_ERROR = None


# --- CONFIGURATION ---
CAM_IDX = 0
AUTO_SCAN_CAMERAS = True
CAMERA_SCAN_LIMIT = 6
REQUESTED_WIDTH, REQUESTED_HEIGHT = 640, 480
REQUESTED_FPS = 60          # OV9782 supports 60 at this resolution. Camera
                            # may not honor the request - actual FPS is
                            # printed at startup.
CROP_SIZE = 96
RECORD_SECONDS = 60  # Wall-clock seconds of actual recording. Timer starts
                     # AFTER calibration completes, so calibration time
                     # doesn't eat into the budget.
VIDEO_WRITER_FPS_FALLBACK = 30.0

# Calibration / deployment
CALIBRATION_FILE = "eye_roi_calibration.json"
HEADLESS_MODE = False        # Research: False. Future Pi deployment: True.
CALIBRATE_EVERY_SESSION = True   # Research default: always recalibrate. Rig
                                 # slippage between sessions is the norm, not
                                 # the exception. Headless mode ignores this
                                 # and uses the saved JSON.
INITIAL_SOURCE_SIZE = 320    # Default ROI in pixels. Big enough for full gaze range.
SHOW_DEBUG_WINDOWS = False   # Runtime preview during recording. Default off
                             # because cv2.imshow on macOS is slow and halves
                             # FPS. The calibration UI is unaffected by this -
                             # it always shows its window when needed.

# Neon
NEON_IP = "192.168.1.9"
NEON_PORT = 8080
AUTO_DISCOVER_NEON = False
NEON_DISCOVERY_SECONDS = 5.0
NEON_GAZE_WIDTH = 1600.0
NEON_GAZE_HEIGHT = 1200.0

# Output files
ORIGINAL_VIDEO = "final_original_dataset.avi"
PROCESSED_VIDEO = "final_96x96_dataset.avi"
CAMERA_FRAMES_CSV = "camera_frames.csv"    # Every frame, no sync filtering
NEON_GAZE_CSV = "neon_gaze_raw.csv"        # Every gaze sample from Neon
DATA_CSV = "final_synced_labels.csv"       # Written by sync_offline.py (not here)


BACKEND_NAMES = {
    getattr(cv2, name): name
    for name in ("CAP_ANY", "CAP_DSHOW", "CAP_MSMF")
    if hasattr(cv2, name)
}


stop_event = threading.Event()
recalibrate_event = threading.Event()
camera_frames_log = []   # one row per camera frame, no filtering
neon_gaze_log = []       # one row per Neon gaze sample

neon_log_lock = threading.Lock()
neon_lock = threading.Lock()
latest_neon_data = {"ts": None, "adjusted_ts": None, "x": None, "y": None, "worn": None}
neon_status = {"connected": False, "error": None, "device_name": None}


# ---------- Data types ----------

@dataclass
class TrackingResult:
    crop: np.ndarray
    debug_mask: np.ndarray
    center: tuple[int, int]
    detected: bool


@dataclass
class ROICalibration:
    center_x: int
    center_y: int
    source_size: int

    def to_dict(self) -> dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "source_size": self.source_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ROICalibration":
        return cls(
            center_x=int(data["center_x"]),
            center_y=int(data["center_y"]),
            source_size=int(data["source_size"]),
        )


# ---------- Cropping ----------

def crop_square(frame, center, source_size, output_size):
    """Crop a square around center, zero-pad if off-frame, resize to output_size."""
    height, width = frame.shape[:2]
    radius = source_size // 2
    cx, cy = center

    x1, y1 = cx - radius, cy - radius
    x2, y2 = cx + radius, cy + radius

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    safe_x1 = max(0, x1)
    safe_y1 = max(0, y1)
    safe_x2 = min(width, x2)
    safe_y2 = min(height, y2)

    crop = frame[safe_y1:safe_y2, safe_x1:safe_x2]
    if crop.size == 0:
        crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    elif pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    if crop.shape[0] != output_size or crop.shape[1] != output_size:
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop


class FixedROICropper:
    """One calibration, fixed slice every frame. Cheap enough for a Pi."""

    def __init__(self, crop_size, calibration):
        self.crop_size = crop_size
        self.calibration = calibration

    def extract(self, frame):
        cal = self.calibration
        center = (cal.center_x, cal.center_y)
        crop = crop_square(frame, center, cal.source_size, self.crop_size)

        debug = frame.copy()
        radius = cal.source_size // 2
        h, w = frame.shape[:2]
        x1 = max(0, cal.center_x - radius)
        y1 = max(0, cal.center_y - radius)
        x2 = min(w - 1, cal.center_x + radius)
        y2 = min(h - 1, cal.center_y + radius)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(debug, center, 3, (0, 255, 0), -1)

        return TrackingResult(crop=crop, debug_mask=debug, center=center, detected=True)


# ---------- Calibration UI ----------

def calibrate_roi(cap, crop_size=CROP_SIZE, initial_source_size=INITIAL_SOURCE_SIZE,
                  initial_center=None, min_size=128, max_size=560, step=12):
    """Interactive click-to-calibrate with live 96x96 preview. Returns None if cancelled.

    If initial_center is provided, the calibration starts with that center
    already set (as if the user clicked there). They can fine-tune or click
    elsewhere.

    User should roll eyes through full gaze range during this step. The pupil
    must stay inside the green box at every extreme. If it leaves, the box is
    too small - press + to grow it.
    """
    state = {"center": initial_center, "source_size": initial_source_size}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["center"] = (x, y)

    window_main = "Calibrate Eye ROI"
    window_preview = "Final 96x96 Preview"
    cv2.namedWindow(window_main)
    cv2.namedWindow(window_preview)
    cv2.setMouseCallback(window_main, on_mouse)

    instructions = [
        "1. Click the eye center",
        "2. Look UP / DOWN / LEFT / RIGHT",
        "3. Pupil must stay in green box",
        "+/-: resize   WASD: nudge",
        "ENTER: save   ESC: cancel",
    ]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        display = frame.copy()
        h, w = frame.shape[:2]

        center = state["center"] if state["center"] else (w // 2, h // 2)
        radius = state["source_size"] // 2
        x1 = max(0, center[0] - radius)
        y1 = max(0, center[1] - radius)
        x2 = min(w - 1, center[0] + radius)
        y2 = min(h - 1, center[1] + radius)

        color = (0, 255, 0) if state["center"] else (0, 200, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.circle(display, center, 4, color, -1)

        for i, txt in enumerate(instructions):
            cv2.putText(display, txt, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, "size=" + str(state["source_size"]) + "px",
                    (10, 22 + len(instructions) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Live 96x96 preview so the user sees what the model will actually see
        preview_crop = crop_square(frame, center, state["source_size"], crop_size)
        preview_enlarged = cv2.resize(preview_crop, (384, 384), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(window_main, display)
        cv2.imshow(window_preview, preview_enlarged)

        key = cv2.waitKey(20) & 0xFF
        if key == 13 and state["center"]:
            break
        elif key == 27:
            cv2.destroyWindow(window_main)
            cv2.destroyWindow(window_preview)
            return None
        elif key in (ord("+"), ord("=")):
            state["source_size"] = min(max_size, state["source_size"] + step)
        elif key in (ord("-"), ord("_")):
            state["source_size"] = max(min_size, state["source_size"] - step)
        elif state["center"] is not None:
            cx, cy = state["center"]
            if key == ord("w"):
                state["center"] = (cx, max(0, cy - 2))
            elif key == ord("s"):
                state["center"] = (cx, min(h - 1, cy + 2))
            elif key == ord("a"):
                state["center"] = (max(0, cx - 2), cy)
            elif key == ord("d"):
                state["center"] = (min(w - 1, cx + 2), cy)

    cv2.destroyWindow(window_main)
    cv2.destroyWindow(window_preview)
    cx, cy = state["center"]
    return ROICalibration(center_x=cx, center_y=cy, source_size=state["source_size"])


def save_calibration(cal, path=CALIBRATION_FILE):
    with open(path, "w") as f:
        json.dump(cal.to_dict(), f, indent=2)


def load_calibration(path=CALIBRATION_FILE):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return ROICalibration.from_dict(json.load(f))
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def get_or_calibrate(cap, crop_size, calibrate_every_session=True, headless=False,
                     calibration_path=CALIBRATION_FILE):
    """Get a cropper, either by loading saved calibration or running the UI.

    Research flow (headless=False, calibrate_every_session=True):
        Always run the calibration UI. Preload prior values as starting
        point so the user doesn't start from zero each session.

    Pi flow (headless=True):
        Load saved JSON. Missing file = hard error. No UI ever shown.
        TODO: replace with auto/button calibration for real deployment.
    """
    previous = load_calibration(calibration_path)

    if headless:
        if previous is None:
            raise RuntimeError(
                "HEADLESS_MODE is enabled but no calibration file was found at '"
                + calibration_path + "'. Run this script once on a machine with a "
                "display to generate it, then copy the JSON to the device. "
                "TODO: replace with auto/button calibration for real deployment."
            )
        print(
            "[Cropper] Headless mode. Loaded calibration from " + calibration_path + ": "
            "center=(" + str(previous.center_x) + "," + str(previous.center_y) + "), "
            "size=" + str(previous.source_size)
        )
        return FixedROICropper(crop_size, previous)

    # UI mode
    if not calibrate_every_session and previous is not None:
        print(
            "[Cropper] Loaded calibration from " + calibration_path + ": "
            "center=(" + str(previous.center_x) + "," + str(previous.center_y) + "), "
            "size=" + str(previous.source_size)
        )
        return FixedROICropper(crop_size, previous)

    if previous is not None:
        print(
            "[Cropper] Running calibration UI (preloaded prior: center=("
            + str(previous.center_x) + "," + str(previous.center_y) + "), size="
            + str(previous.source_size) + ")"
        )
        cal = calibrate_roi(
            cap,
            crop_size=crop_size,
            initial_center=(previous.center_x, previous.center_y),
            initial_source_size=previous.source_size,
        )
    else:
        print("[Cropper] Running calibration UI (no prior calibration found)...")
        cal = calibrate_roi(cap, crop_size=crop_size)

    if cal is None:
        # User cancelled. Fall back to prior if we have one.
        if previous is not None:
            print("[Cropper] Calibration cancelled. Using previous saved calibration.")
            return FixedROICropper(crop_size, previous)
        raise RuntimeError("Calibration cancelled and no previous calibration exists.")

    save_calibration(cal, calibration_path)
    print(
        "[Cropper] Saved calibration to " + calibration_path + ": "
        "center=(" + str(cal.center_x) + "," + str(cal.center_y) + "), "
        "size=" + str(cal.source_size)
    )
    return FixedROICropper(crop_size, cal)


# ---------- Camera ----------

def camera_backends():
    if os.name == "nt":
        backends = [
            backend
            for backend in (
                getattr(cv2, "CAP_DSHOW", None),
                getattr(cv2, "CAP_MSMF", None),
                getattr(cv2, "CAP_ANY", None),
            )
            if backend is not None
        ]
        return list(dict.fromkeys(backends))
    return [cv2.CAP_ANY]


def warmup_camera(cap):
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        time.sleep(0.05)
    return None


def get_capture_fps(cap):
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1.0 or fps > 240.0:
        return VIDEO_WRITER_FPS_FALLBACK
    return fps


def open_camera():
    candidate_indices = [CAM_IDX]
    if AUTO_SCAN_CAMERAS:
        candidate_indices.extend(
            idx for idx in range(CAMERA_SCAN_LIMIT) if idx != CAM_IDX
        )

    for camera_index in candidate_indices:
        for backend in camera_backends():
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

            frame = warmup_camera(cap)
            if frame is None:
                cap.release()
                continue

            actual_height, actual_width = frame.shape[:2]
            capture_fps = get_capture_fps(cap)
            backend_name = BACKEND_NAMES.get(backend, str(backend))
            print(
                "[Camera] Opened camera index " + str(camera_index) + " via " + backend_name
                + " at " + str(actual_width) + "x" + str(actual_height)
                + ". Reported FPS: " + ("%.1f" % capture_fps)
            )
            return cap, camera_index, frame, capture_fps

    return None, None, None, None


def create_writer(path, writer_fps, frame_size):
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        writer_fps,
        frame_size,
    )
    if writer.isOpened():
        return writer
    writer.release()
    print("[Writer] Could not create '" + path + "'.")
    return None


# ---------- Neon ----------

def get_neon_snapshot():
    with neon_lock:
        return {
            "ts": latest_neon_data["ts"],
            "adjusted_ts": latest_neon_data["adjusted_ts"],
            "x": latest_neon_data["x"],
            "y": latest_neon_data["y"],
            "worn": latest_neon_data["worn"],
            "connected": neon_status["connected"],
            "error": neon_status["error"],
            "device_name": neon_status["device_name"],
        }


def is_tcp_endpoint_reachable(address, port, timeout_seconds=2.0):
    try:
        with socket.create_connection((address, port), timeout=timeout_seconds):
            return True, None
    except OSError as exc:
        return False, str(exc)


def connect_neon():
    if NeonDevice is None:
        with neon_lock:
            neon_status["error"] = (
                "Missing dependency 'pupil-labs-realtime-api'. "
                "Install it with: pip install pupil-labs-realtime-api"
            )
        return None

    if AUTO_DISCOVER_NEON and discover_one_device is not None:
        print("[Neon] Looking for a device on the local network...")
        try:
            device = discover_one_device(max_search_duration_seconds=NEON_DISCOVERY_SECONDS)
            if device is not None:
                return device
        except Exception as exc:
            print("[Neon] Auto-discovery failed: " + str(exc))

    if NEON_IP:
        print("[Neon] Trying configured address " + NEON_IP + ":" + str(NEON_PORT) + "...")
        is_reachable, reachability_error = is_tcp_endpoint_reachable(NEON_IP, NEON_PORT)
        if not is_reachable:
            with neon_lock:
                neon_status["error"] = (
                    "Neon not reachable at " + NEON_IP + ":" + str(NEON_PORT) + ". "
                    "Socket check failed: " + str(reachability_error) + ". "
                    "Make sure the PC and Neon are on the same network, "
                    "the IP is correct, and realtime streaming is enabled."
                )
            return None

        if not hasattr(NeonDevice, "_event_manager"):
            NeonDevice._event_manager = None
        return NeonDevice(address=NEON_IP, port=NEON_PORT)

    return None


def neon_loop():
    device = None
    clock_offset_seconds = None
    try:
        device = connect_neon()
        if device is None:
            return

        device_name = (
            getattr(device, "serial_number_glasses", None)
            or getattr(device, "full_name", None)
            or (NEON_IP + ":" + str(NEON_PORT))
        )
        with neon_lock:
            neon_status.update({"connected": True, "error": None, "device_name": device_name})
        print("[Neon] Connected to " + str(device_name) + ".")

        while not stop_event.is_set():
            gaze = device.receive_gaze_datum(timeout_seconds=0.5)
            if gaze is None:
                continue

            host_now = time.time()
            raw_ts = float(gaze.timestamp_unix_seconds)
            observed_offset = host_now - raw_ts
            if clock_offset_seconds is None:
                clock_offset_seconds = observed_offset
            else:
                clock_offset_seconds = clock_offset_seconds * 0.98 + observed_offset * 0.02

            with neon_lock:
                latest_neon_data.update(
                    {
                        "ts": raw_ts,
                        "adjusted_ts": raw_ts + clock_offset_seconds,
                        "x": float(gaze.x),
                        "y": float(gaze.y),
                        "worn": bool(getattr(gaze, "worn", True)),
                    }
                )

            with neon_log_lock:
                neon_gaze_log.append(
                    [
                        raw_ts,
                        raw_ts + clock_offset_seconds,
                        float(gaze.x),
                        float(gaze.y),
                        bool(getattr(gaze, "worn", True)),
                    ]
                )
    except Exception as exc:
        with neon_lock:
            neon_status["connected"] = False
            neon_status["error"] = str(exc)
        print("[Neon] Error: " + str(exc))
    finally:
        with neon_lock:
            neon_status["connected"] = False
        if device is not None:
            with contextlib.suppress(Exception):
                device.close()


# ---------- Preview + main loop ----------

def draw_debug_windows(tracking):
    try:
        cv2.imshow("Real-Time 96x96 Crop", tracking.crop)
        cv2.imshow("ROI Preview", tracking.debug_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
        elif key == ord("r"):
            print("[Cropper] Recalibration requested.")
            recalibrate_event.set()
        return True
    except cv2.error as exc:
        print("[UI] Debug windows disabled: " + str(exc))
        return False


def arducam_loop():
    cap, active_index, first_frame, capture_fps = open_camera()
    if cap is None:
        print("[Camera] Could not open any camera. Check CAM_IDX and close other camera apps.")
        stop_event.set()
        return

    try:
        cropper = get_or_calibrate(
            cap, CROP_SIZE,
            calibrate_every_session=CALIBRATE_EVERY_SESSION,
            headless=HEADLESS_MODE,
        )
    except RuntimeError as exc:
        print("[Cropper] " + str(exc))
        stop_event.set()
        cap.release()
        return

    writer_fps = capture_fps or VIDEO_WRITER_FPS_FALLBACK
    processed_writer = create_writer(PROCESSED_VIDEO, writer_fps, (CROP_SIZE, CROP_SIZE))
    original_writer = create_writer(
        ORIGINAL_VIDEO, writer_fps, (first_frame.shape[1], first_frame.shape[0])
    )
    if processed_writer is None or original_writer is None:
        stop_event.set()
        cap.release()
        if processed_writer is not None:
            processed_writer.release()
        if original_writer is not None:
            original_writer.release()
        return

    frame_idx = 0
    show_debug = SHOW_DEBUG_WINDOWS and not HEADLESS_MODE
    next_status_print = 0.0

    print("[Camera] Cropper ready on camera index " + str(active_index) + ". Recording starting...")

    # Start the recording timer NOW, after calibration finishes
    record_deadline = time.time() + RECORD_SECONDS

    try:
        while not stop_event.is_set():
            if time.time() >= record_deadline:
                stop_event.set()
                break
            if recalibrate_event.is_set() and not HEADLESS_MODE:
                recalibrate_event.clear()
                try:
                    cropper = get_or_calibrate(
                        cap, CROP_SIZE,
                        calibrate_every_session=True,
                        headless=False,
                    )
                except RuntimeError as exc:
                    print("[Cropper] Recalibration failed: " + str(exc) + ". Keeping previous calibration.")

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            cam_ts = time.time()
            tracking = cropper.extract(frame)
            original_writer.write(frame)
            processed_writer.write(tracking.crop)

            camera_frames_log.append(
                [
                    frame_idx,
                    cam_ts,
                    tracking.center[0],
                    tracking.center[1],
                    cropper.calibration.source_size,
                ]
            )
            frame_idx += 1

            if time.time() >= next_status_print:
                neon = get_neon_snapshot()
                with neon_log_lock:
                    gaze_samples = len(neon_gaze_log)
                if neon["connected"]:
                    print(
                        "[Status] Frames: " + str(frame_idx)
                        + "  Neon samples: " + str(gaze_samples)
                    )
                elif neon["error"]:
                    print("[Status] Recording frames. Neon unavailable: " + str(neon["error"]))
                else:
                    print("[Status] Recording frames. Waiting for Neon connection.")
                next_status_print = time.time() + 3.0

            if show_debug:
                show_debug = draw_debug_windows(tracking)
    finally:
        cap.release()
        if processed_writer is not None:
            processed_writer.release()
        if original_writer is not None:
            original_writer.release()
        cv2.destroyAllWindows()


def save_dataset():
    wrote_anything = False

    if camera_frames_log:
        columns = ["frame", "cam_ts", "crop_cx", "crop_cy", "crop_source_size"]
        cam_df = pd.DataFrame(camera_frames_log, columns=columns)
        cam_df.to_csv(CAMERA_FRAMES_CSV, index=False)
        print("[Save] Wrote " + str(len(cam_df)) + " camera frames to '" + CAMERA_FRAMES_CSV + "'.")
        wrote_anything = True
    else:
        print("[Save] No camera frames captured.")

    with neon_log_lock:
        neon_snapshot = list(neon_gaze_log)

    if neon_snapshot:
        columns = ["ts", "adjusted_ts", "gaze_x", "gaze_y", "worn"]
        neon_df = pd.DataFrame(neon_snapshot, columns=columns)
        neon_df.to_csv(NEON_GAZE_CSV, index=False)
        print("[Save] Wrote " + str(len(neon_df)) + " Neon gaze samples to '" + NEON_GAZE_CSV + "'.")
        wrote_anything = True
    else:
        print("[Save] No Neon gaze samples captured.")

    if wrote_anything:
        print("\n[Next step] Run 'python sync_offline.py' to produce '" + DATA_CSV + "'.")
    return wrote_anything


def main():
    stop_event.clear()
    recalibrate_event.clear()
    camera_frames_log.clear()
    with neon_log_lock:
        neon_gaze_log.clear()

    if PUPIL_LABS_IMPORT_ERROR is not None:
        print(
            "[Setup] Neon support unavailable because "
            "'pupil-labs-realtime-api' is not installed: " + str(PUPIL_LABS_IMPORT_ERROR)
        )

    neon_thread = None
    if NeonDevice is not None:
        neon_thread = threading.Thread(target=neon_loop, daemon=True)
        neon_thread.start()
        time.sleep(1.0)

    print(
        "\n[Recorder] " + str(RECORD_SECONDS) + " seconds of recording will start "
        "after calibration. Press 'q' in the preview to stop, 'r' to recalibrate, "
        "Ctrl+C to abort."
    )

    try:
        arducam_loop()
    except KeyboardInterrupt:
        print("\n[Recorder] Keyboard interrupt received.")
    finally:
        print("[Recorder] Stopping...")
        stop_event.set()
        if neon_thread is not None:
            neon_thread.join(timeout=2.0)
        save_dataset()


if __name__ == "__main__":
    main()