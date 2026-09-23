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
            presets = {"hybrid_baritone", "dark_sarcastic"}
            if self.default_voice not in available_voices and self.default_voice not in presets:
                print(f"[ULTRON TTS] Warning: Voice '{self.default_voice}' not recognized.")
                self.default_voice = "am_michael" if "am_michael" in available_voices else available_voices[0]

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

    def _resolve_voice(self, voice_spec: str | np.ndarray) -> str | np.ndarray:
        """Resolve voice name, blend preset, or raw vector."""
        if isinstance(voice_spec, np.ndarray):
            return voice_spec

        name = (voice_spec or self.default_voice).strip().lower()

        if name == "hybrid_baritone":
            # 50% Onyx (baritone depth) + 50% Michael (confident spoken realism)
            v_onyx = self._kokoro.get_voice_style("am_onyx")
            v_mike = self._kokoro.get_voice_style("am_michael")
            return 0.5 * v_onyx + 0.5 * v_mike

        if name == "dark_sarcastic":
            # 60% Fenrir (deep menace) + 40% Puck (witty sarcasm)
            v_fen = self._kokoro.get_voice_style("am_fenrir")
            v_puck = self._kokoro.get_voice_style("am_puck")
            return 0.6 * v_fen + 0.4 * v_puck

        if name in self._kokoro.get_voices():
            return name

        # Default fallback
        available = self._kokoro.get_voices()
        return "am_michael" if "am_michael" in available else available[0]

    def _synthesize_edge(self, text: str) -> tuple[np.ndarray, int, float]:
        """Synthesize via Edge-TTS (Neural Christopher) for realistic human emotion."""
        import asyncio
        import io
        import soundfile as sf
        import edge_tts
        import concurrent.futures

        start = time.perf_counter()
        voice = getattr(config, "TTS_EDGE_VOICE", "en-US-ChristopherNeural")
        rate = getattr(config, "TTS_EDGE_RATE", "+0%")
        pitch = getattr(config, "TTS_EDGE_PITCH", "-3Hz")

        async def _run_edge():
            comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
            buf = io.BytesIO()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            data, sr = sf.read(buf, dtype="float32")
            return data, sr

        try:
            # Safely handle async execution from any thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                audio, sample_rate = pool.submit(asyncio.run, _run_edge()).result(timeout=6.0)

            latency_ms = (time.perf_counter() - start) * 1000
            return audio, sample_rate, latency_ms

        except Exception as e:
            print(f"[ULTRON TTS] Edge-TTS error ({e}), falling back to local Kokoro...")
            return self._synthesize_kokoro(text)

    def _synthesize_kokoro(
        self,
        text: str,
        voice: str | np.ndarray | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Local offline Kokoro synthesis fallback."""
        if not self._is_loaded:
            if not self.load():
                return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0

        target_voice = self._resolve_voice(voice or self.default_voice)
        target_speed = speed if speed is not None else self.default_speed
        sent_pause = getattr(config, "TTS_SENTENCE_PAUSE", 0.12)
        clause_pause = getattr(config, "TTS_CLAUSE_PAUSE", 0.06)

        start = time.perf_counter()
        try:
            with self._lock:
                audio, sample_rate = self._kokoro.create(
                    text,
                    voice=target_voice,
                    speed=target_speed,
                    sentence_pause=sent_pause,
                    clause_pause=clause_pause,
                    lang="en-us",
                )

            latency_ms = (time.perf_counter() - start) * 1000
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            return audio, sample_rate, latency_ms

        except Exception as e:
            print(f"[ULTRON TTS] Kokoro error synthesizing text \"{text[:40]}...\": {e}")
            return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0

    def synthesize(
        self,
        text: str,
        voice: str | np.ndarray | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Synthesize speech audio from text.

        Uses Edge-TTS Neural Christopher when configured (for deep emotion & human swagger)
        with automatic fallback to local Kokoro ONNX.
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0

        provider = getattr(config, "TTS_PROVIDER", "edge")
        if provider == "edge":
            return self._synthesize_edge(cleaned_text)
        else:
            return self._synthesize_kokoro(cleaned_text, voice, speed)
