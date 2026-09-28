"""
ULTRON Security — Camera Tampering & Obstruction Detector
=========================================================
Deterministic vision analysis to detect physical camera tampering:
  1. Lens Covering / Blackout: Hand, cloth, or tape placed over camera.
     Detected via near-zero mean brightness or flat standard deviation.
  2. Defocus / Smear / Spray: Heavy blurring or spray paint on the lens.
     Detected via severe collapse in Laplacian edge variance.

Debounced over consecutive frames to prevent false alarms from transient
lighting shifts or camera exposure adjustments.
"""

import time
import cv2
import numpy as np

import config
from core.event_bus import EventBus, EventTypes


class CameraTamperDetector:
    """
    Analyzes camera frames for physical tampering or optical obstruction.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self.min_frames = getattr(config, "TAMPER_MIN_FRAMES", 15)
        self.blackout_thresh = getattr(config, "TAMPER_BLACKOUT_BRIGHTNESS", 18.0)
        self.flatness_thresh = getattr(config, "TAMPER_BLACKOUT_STD", 8.0)
        self.blur_thresh = getattr(config, "TAMPER_BLUR_LAPLACIAN_VAR", 22.0)

        self._consecutive_tamper = 0
        self._consecutive_clear = 0
        self._is_obstructed = False
        self._last_obstruction_type = ""

    @property
    def is_obstructed(self) -> bool:
        """True if camera is actively confirmed to be obstructed."""
        return self._is_obstructed

    def evaluate_frame(self, frame_bgr: np.ndarray) -> tuple[bool, str, dict]:
        """
        Evaluate a single BGR camera frame for tampering.

        Returns:
            (is_tampered, tamper_type, metrics_dict)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, "", {}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        metrics = {
            "mean_brightness": mean_val,
            "std_deviation": std_val,
            "laplacian_var": laplacian_var,
        }

        # Check conditions
        detected_type = ""
        if mean_val < self.blackout_thresh or std_val < self.flatness_thresh:
            detected_type = "LENS_COVERED"
        elif laplacian_var < self.blur_thresh and mean_val > self.blackout_thresh:
            detected_type = "DEFOCUS_OR_SPRAY"

        if detected_type:
            self._consecutive_tamper += 1
            self._consecutive_clear = 0

            # Trigger only once threshold is reached
            if self._consecutive_tamper >= self.min_frames and not self._is_obstructed:
                self._is_obstructed = True
                self._last_obstruction_type = detected_type
                print(f"[ULTRON Security] CAMERA TAMPER CONFIRMED: {detected_type} (metrics: {metrics})")

                self.event_bus.publish(EventTypes.CAMERA_OBSTRUCTED, {
                    "type": detected_type,
                    "metrics": metrics,
                    "timestamp": time.time(),
                })
        else:
            self._consecutive_clear += 1
            self._consecutive_tamper = 0

            # Clear obstruction if clear for threshold frames
            if self._consecutive_clear >= self.min_frames and self._is_obstructed:
                self._is_obstructed = False
                prev_type = self._last_obstruction_type
                self._last_obstruction_type = ""
                print(f"[ULTRON Security] Camera tamper cleared ({prev_type}).")

                self.event_bus.publish(EventTypes.TAMPER_CLEARED, {
                    "cleared_type": prev_type,
                    "timestamp": time.time(),
                })

        return self._is_obstructed, self._last_obstruction_type, metrics
