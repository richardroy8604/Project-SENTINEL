"""
ULTRON Vision — Camera Capture Module
======================================
Threaded webcam capture using OpenCV.
Provides the latest frame via a thread-safe buffer (drop-oldest pattern)
so the UI never lags behind real-time, even if processing is slow.
"""

import threading
import time
import cv2
import numpy as np

import config


class Camera:
    """
    Threaded webcam capture.

    Usage:
        cam = Camera()
        cam.start()
        frame = cam.get_frame()  # Always returns the latest frame (or None)
        cam.stop()
    """

    def __init__(
        self,
        device_index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        target_fps: int = config.CAMERA_FPS,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps

        # Thread-safe frame buffer
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Performance tracking
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._last_fps_time: float = 0.0

    def start(self) -> bool:
        """Start the capture thread. Returns True if camera opened successfully."""
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            print(f"[ULTRON Vision] ERROR: Could not open camera {self.device_index}")
            return False

        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Read actual resolution (camera may not support requested size)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[ULTRON Vision] Camera opened: {actual_w}x{actual_h}")

        # Update config if camera gave us a different resolution
        self.width = actual_w
        self.height = actual_h

        self._running = True
        self._last_fps_time = time.perf_counter()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def _capture_loop(self):
        """Background thread: continuously reads frames from the webcam."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Store the latest frame (drop-oldest: overwrites previous)
            with self._lock:
                self._frame = frame

            # FPS calculation
            self._frame_count += 1
            now = time.perf_counter()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

    def get_frame(self) -> np.ndarray | None:
        """
        Get the latest captured frame.

        Returns:
            BGR numpy array (H, W, 3) or None if no frame available yet.
        """
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_frame_rgb(self) -> np.ndarray | None:
        """
        Get the latest frame converted to RGB.
        Useful for display frameworks that expect RGB format.
        """
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    @property
    def fps(self) -> float:
        """Current capture FPS."""
        return self._fps

    @property
    def is_running(self) -> bool:
        """Whether the capture thread is active."""
        return self._running

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
        print("[ULTRON Vision] Camera released.")
