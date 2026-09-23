"""
ULTRON Voice — Dual TTS Engine (Edge Neural + Kokoro Local)
============================================================
Provides high-expressiveness neural speech synthesis:
  - Edge Neural TTS (Primary): State-of-the-art human emotion, dramatic pauses,
    natural inflection, and vocal fry using 'en-US-ChristopherNeural' (pitch: -8Hz).
  - Kokoro ONNX (Fallback): 100% offline local model.
  - Optional Ultron Metallic FX: Subtle synthetic comb-filter presence.
"""

import asyncio
import io
import re
import threading
import time
from pathlib import Path
import numpy as np
import soundfile as sf

import config


class TTSEngine:
    """
    Speech synthesis engine supporting Edge Neural TTS and Kokoro ONNX.

    Usage:
        tts = TTSEngine()
        if tts.load():
            audio, sr, latency = tts.synthesize("Smile. You're on camera.")
    """

    def __init__(
        self,
        provider: str = config.TTS_PROVIDER,
        edge_voice: str = config.TTS_EDGE_VOICE,
        edge_pitch: str = config.TTS_EDGE_PITCH,
        edge_rate: str = config.TTS_EDGE_RATE,
        model_path: Path = config.TTS_MODEL_PATH,
        voices_path: Path = config.TTS_VOICES_PATH,
        default_voice: str = config.TTS_VOICE,
        default_speed: float = config.TTS_SPEED,
    ):
        self.provider = provider
        self.edge_voice = edge_voice
        self.edge_pitch = edge_pitch
        self.edge_rate = edge_rate

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
        """Initialize TTS provider."""
        if self._is_loaded:
            return True

        # If provider is Edge Neural TTS, test import
        if self.provider == "edge":
            try:
                import edge_tts
                self._is_loaded = True
                print(f"[ULTRON TTS] Edge Neural TTS ready -- Voice: '{self.edge_voice}' (Pitch: {self.edge_pitch})")
                return True
            except ImportError:
                print("[ULTRON TTS] edge-tts package not installed, falling back to Kokoro.")
                self.provider = "kokoro"

        # Kokoro Local Engine (or Fallback)
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
        - Removes stage directions like [laughs], (whispering)
        - Strips prefix headers like 'ULTRON:' or 'Assistant:'
        - Normalizes multiple spaces while PRESERVING ellipses (...) and dashes (--)
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

        # Clean excess spaces
        t = re.sub(r"[ \t]+", " ", t).strip()

        return t

    def _resolve_kokoro_voice(self, voice_spec: str | np.ndarray) -> str | np.ndarray:
        """Resolve voice name, blend preset, or raw vector for Kokoro."""
        if isinstance(voice_spec, np.ndarray):
            return voice_spec

        name = (voice_spec or self.default_voice).strip().lower()

        if name == "hybrid_baritone":
            v_onyx = self._kokoro.get_voice_style("am_onyx")
            v_mike = self._kokoro.get_voice_style("am_michael")
            return 0.5 * v_onyx + 0.5 * v_mike

        if name == "dark_sarcastic":
            v_fen = self._kokoro.get_voice_style("am_fenrir")
            v_puck = self._kokoro.get_voice_style("am_puck")
            return 0.6 * v_fen + 0.4 * v_puck

        if name in self._kokoro.get_voices():
            return name

        available = self._kokoro.get_voices()
        return "am_michael" if "am_michael" in available else available[0]

    def _apply_ultron_fx(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply subtle synthetic metallic acoustic resonance to give ULTRON presence."""
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        delay_samples = int(sample_rate * 0.016)  # 16ms reflection
        if len(audio) <= delay_samples:
            return audio
        wet = np.zeros_like(audio)
        wet[delay_samples:] = audio[:-delay_samples]
        filtered = 0.82 * audio + 0.28 * wet
        peak = np.max(np.abs(filtered))
        if peak > 0:
            filtered = (filtered / peak) * 0.95
        return filtered.astype(np.float32)

    def _synthesize_edge_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous wrapper for in-memory Edge-TTS synthesis."""
        import edge_tts

        async def _run():
            comm = edge_tts.Communicate(
                text,
                self.edge_voice,
                rate=self.edge_rate,
                pitch=self.edge_pitch,
            )
            buf = io.BytesIO()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            audio, sr = sf.read(buf)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            return audio.astype(np.float32), sr

        try:
            return asyncio.run(_run())
        except RuntimeError:
            # If an event loop is already running in this thread
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_run())
            finally:
                loop.close()

    def synthesize(
        self,
        text: str,
        voice: str | np.ndarray | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Synthesize speech audio from text with natural conversational cadence.

        Returns:
            (audio_samples_float32, sample_rate, latency_ms)
        """
        if not self._is_loaded:
            if not self.load():
                return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0

        start = time.perf_counter()

        # 1. Primary: Edge Neural TTS (Dramatic pauses, human inflection, vocal fry)
        if self.provider == "edge":
            try:
                audio, sample_rate = self._synthesize_edge_sync(cleaned_text)
                if getattr(config, "TTS_ULTRON_FX", False):
                    audio = self._apply_ultron_fx(audio, sample_rate)
                latency_ms = (time.perf_counter() - start) * 1000
                return audio, sample_rate, latency_ms
            except Exception as e:
                print(f"[ULTRON TTS] Edge TTS error: {e}. Falling back to Kokoro...")

        # 2. Fallback: Kokoro Local ONNX
        if self._kokoro is None:
            self.load()

        if self._kokoro is not None:
            try:
                target_voice = self._resolve_kokoro_voice(voice or self.default_voice)
                target_speed = speed if speed is not None else self.default_speed
                sent_pause = getattr(config, "TTS_SENTENCE_PAUSE", 0.12)
                clause_pause = getattr(config, "TTS_CLAUSE_PAUSE", 0.06)

                with self._lock:
                    audio, sample_rate = self._kokoro.create(
                        cleaned_text,
                        voice=target_voice,
                        speed=target_speed,
                        sentence_pause=sent_pause,
                        clause_pause=clause_pause,
                        lang="en-us",
                    )

                if getattr(config, "TTS_ULTRON_FX", False):
                    audio = self._apply_ultron_fx(audio, sample_rate)

                latency_ms = (time.perf_counter() - start) * 1000
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                return audio, sample_rate, latency_ms

            except Exception as e:
                print(f"[ULTRON TTS] Kokoro synthesis error: {e}")

        return np.array([], dtype=np.float32), config.TTS_SAMPLE_RATE, 0.0
