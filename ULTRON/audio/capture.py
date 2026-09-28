"""
ULTRON Audio — Microphone Capture Module
=========================================
Captures real-time mono microphone input at 16 kHz using sounddevice (WASAPI).
Computes real-time RMS audio levels (0.0 to 1.0) for the UI VU meter and forwards
audio frames to the VAD and Speech pipeline.
"""

import threading
import time
from typing import Callable
import numpy as np
import sounddevice as sd

import config


class AudioCapture:
    """
    Threaded microphone input stream.

    Usage:
        capture = AudioCapture(
            on_frame=vad_processor.process_frame,
            on_level=dashboard.update_audio_level
        )
        capture.start()
        ...
        capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        block_size: int = config.AUDIO_BLOCK_SIZE,
        on_frame: Callable[[np.ndarray], None] | None = None,
        on_level: Callable[[float], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.on_frame = on_frame
        self.on_level = on_level

        self._stream: sd.InputStream | None = None
        self._running = False
        self._lock = threading.Lock()
        self._current_level = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_level(self) -> float:
        return self._current_level

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Audio stream callback running in PortAudio/WASAPI thread."""
        if status:
            pass  # Avoid console flooding on buffer overflow

        # indata is shape (frames, channels), float32 [-1.0, 1.0]
        mono = indata[:, 0].copy()

        # Compute RMS energy for VU meter
        rms = np.sqrt(np.mean(mono**2) + 1e-12)
        # Scaled for laptop microphone: audible speech reaches 30-90%
        normalized_level = min(1.0, float(rms * 16.0))
        self._current_level = normalized_level

        if self.on_level is not None:
            try:
                self.on_level(normalized_level)
            except Exception:
                pass

        # Send raw 512-sample chunk to VAD
        if self.on_frame is not None and self._running:
            try:
                self.on_frame(mono)
            except Exception as e:
                print(f"[ULTRON Audio] Frame dispatch error: {e}")

    def start(self) -> bool:
        """Start microphone capture stream."""
        with self._lock:
            if self._running:
                return True

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=config.AUDIO_CHANNELS,
                    dtype="float32",
                    blocksize=self.block_size,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._running = True
                print(f"[ULTRON Audio] Microphone capture active @ {self.sample_rate} Hz (chunk: {self.block_size})")
                return True
            except Exception as e:
                print(f"[ULTRON Audio] ERROR starting microphone stream: {e}")
                self._running = False
                return False

    def stop(self):
        """Stop microphone capture stream."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            print("[ULTRON Audio] Microphone stream stopped.")
