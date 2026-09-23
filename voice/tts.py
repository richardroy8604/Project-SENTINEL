"""
ULTRON Voice — Kokoro TTS ONNX Synthesis Engine
================================================
Generates high-fidelity 24 kHz spoken audio from text using Kokoro-82M ONNX.
Default voice is 'am_onyx' (deep, calm, articulate American male voice).
"""

import re
import threading
import time
from pathlib import Path
import numpy as np

import config


class TTSEngine:
    """
    Kokoro-82M ONNX speech synthesis engine.

    Usage:
        tts = TTSEngine()
        if tts.load():
            audio, sr, latency = tts.synthesize("Smile. You're on camera.")
    """

    def __init__(
        self,
        model_path: Path = config.TTS_MODEL_PATH,
        voices_path: Path = config.TTS_VOICES_PATH,
        default_voice: str = config.TTS_VOICE,
        default_speed: float = config.TTS_SPEED,
    ):
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.default_voice = default_voice
        self.default_speed = default_speed

        self._kokoro = None
        self._is_loaded = False
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> bool:
        """Load Kokoro ONNX model and voice weights into memory."""
        if self._is_loaded:
            return True

        if not self.model_path.exists() or not self.voices_path.exists():
            print(f"[ULTRON TTS] Error: Model files not found. Run setup_models.py first.")
            return False

        try:
            from kokoro_onnx import Kokoro

            print(f"[ULTRON TTS] Loading Kokoro TTS ({self.model_path.name})...")
            start = time.perf_counter()
            self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
            load_time = (time.perf_counter() - start) * 1000

            available_voices = self._kokoro.get_voices()
            if self.default_voice not in available_voices:
                print(f"[ULTRON TTS] Warning: Voice '{self.default_voice}' not found in voices.bin.")
                if available_voices:
                    self.default_voice = available_voices[0]

            self._is_loaded = True
            print(f"[ULTRON TTS] Kokoro TTS loaded in {load_time:.0f}ms -- Voice: '{self.default_voice}'")
            return True

        except Exception as e:
            print(f"[ULTRON TTS] Failed to load Kokoro TTS: {e}")
            self._is_loaded = False
            return False

    def clean_text(self, text: str) -> str:
        """
        Sanitize text for natural speech synthesis:
        - Removes markdown characters (*, _, #, `, ~)
        - Removes stage directions and parentheticals like [laughs], (whispering)
        - Strips prefix headers like 'ULTRON:' or 'Assistant:'
        - Normalizes multiple spaces and punctuation
        """
        if not text:
            return ""

        t = text.strip()

        # Remove prefix tags
        if t.upper().startswith("ULTRON:"):
            t = t[7:].strip()
        elif t.upper().startswith("ASSISTANT:"):
            t = t[10:].strip()

        # Remove quotes wrapping entire sentence
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()

        # Remove bracketed actions: [chuckles], (whispers), etc.
        t = re.sub(r"\[.*?\]", "", t)
        t = re.sub(r"\(.*?\)", "", t)

        # Remove markdown symbols
        t = re.sub(r"[*_~`#>]", "", t)

        # Replace repeated punctuation and clean whitespace
        t = re.sub(r"\s+", " ", t).strip()

        return t

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Synthesize speech audio from text.

        Args:
            text: Dialogue string to synthesize
            voice: Voice ID (defaults to config.TTS_VOICE, e.g. 'am_onyx')
            speed: Speech rate multiplier (defaults to config.TTS_SPEED, e.g. 1.05)

        Returns:
            (audio_samples_float32, sample_rate, latency_ms)
        """
        if not self._is_loaded:
            if not self.load():
                return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0

        target_voice = voice or self.default_voice
        target_speed = speed if speed is not None else self.default_speed

        start = time.perf_counter()
        try:
            with self._lock:
                audio, sample_rate = self._kokoro.create(
                    cleaned_text,
                    voice=target_voice,
                    speed=target_speed,
                    lang="en-us",
                )

            latency_ms = (time.perf_counter() - start) * 1000
            # Ensure float32 format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            return audio, sample_rate, latency_ms

        except Exception as e:
            print(f"[ULTRON TTS] Error synthesizing text \"{cleaned_text[:40]}...\": {e}")
            return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0
