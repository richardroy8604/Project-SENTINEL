"""
ULTRON Audio — Speech-to-Text (faster-whisper)
==============================================
Transcribes spoken audio utterances using faster-whisper (CTranslate2).
Configured to run on CPU with INT8 quantization (0 MB VRAM footprint).
Enforces condition_on_previous_text=False to eliminate hallucination cascades,
and integrates audio/validator.py to discard noise and non-English utterances.
"""

import time
import numpy as np
from faster_whisper import WhisperModel

import config
from audio.validator import validate_english_speech


class SpeechToText:
    """
    faster-whisper STT engine with English validation and noise filtering.

    Usage:
        stt = SpeechToText()
        stt.load()
        text, latency_ms = stt.transcribe(audio_float32_array)
    """

    def __init__(
        self,
        model_size: str = config.STT_MODEL_SIZE,
        device: str = config.STT_DEVICE,
        compute_type: str = config.STT_COMPUTE_TYPE,
        cpu_threads: int = config.STT_CPU_THREADS,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model: WhisperModel | None = None
        self._loaded = False

    def load(self) -> bool:
        """Load faster-whisper model."""
        try:
            print(f"[ULTRON STT] Loading faster-whisper '{self.model_size}' on {self.device} ({self.compute_type})...")
            start = time.perf_counter()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            self._loaded = True
            dur = time.perf_counter() - start
            print(f"[ULTRON STT] faster-whisper loaded in {dur:.2f}s.")
            return True
        except Exception as e:
            print(f"[ULTRON STT] ERROR loading faster-whisper: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio segment into validated English text.

        Args:
            audio: 1D float32 numpy array sampled at 16000 Hz.

        Returns:
            (validated_text, latency_ms) -> validated_text is empty string if noise/rejected.
        """
        if not self._loaded or self._model is None:
            return "", 0.0

        min_samples = int(config.AUDIO_SAMPLE_RATE * (config.VAD_MIN_SPEECH_DURATION_MS / 1000.0))
        if len(audio) < min_samples:
            return "", 0.0

        start_time = time.perf_counter()
        try:
            # Transcribe with anti-hallucination settings:
            # - condition_on_previous_text=False stops hallucination memory loops
            # - beam_size=1 (greedy) provides fast edge performance
            # - strict no_speech and logprob thresholds reject low-confidence noise
            segments, info = self._model.transcribe(
                audio,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
                no_speech_threshold=config.STT_NO_SPEECH_THRESHOLD,
                log_prob_threshold=config.STT_LOG_PROB_THRESHOLD,
                vad_filter=False,  # Audio is already pre-segmented by Silero VAD
            )

            valid_segments = []
            for segment in segments:
                cleaned_text = validate_english_speech(
                    text=segment.text,
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                )
                if cleaned_text:
                    valid_segments.append(cleaned_text)

            full_text = " ".join(valid_segments).strip()
            latency_ms = (time.perf_counter() - start_time) * 1000
            return full_text, latency_ms

        except Exception as e:
            print(f"[ULTRON STT] Transcription error: {e}")
            return "", 0.0
