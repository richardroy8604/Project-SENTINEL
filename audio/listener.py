"""
ULTRON Audio — Audio Listener & Pipeline Orchestrator
======================================================
Connects AudioCapture, Silero VAD, and faster-whisper STT.
Enforces multi-layer filtering:
- Requires sustained vocal frame count (filters out clicks, coughs, desk knocks)
- Requires minimum peak amplitude (filters out faint background hum)
- Validates that decoded text is genuine English without noise hallucinations
- Publishes SPEECH_DETECTED and SPEECH_RECOGNIZED events to the EventBus.
"""

import collections
import queue
import threading
import time
from typing import Callable
import numpy as np

import config
from core.event_bus import EventBus, EventTypes
from audio.capture import AudioCapture
from audio.vad import SileroVAD
from audio.stt import SpeechToText


class AudioListener:
    """
    Complete audio pipeline orchestrator with noise & language filtering.

    Usage:
        listener = AudioListener(event_bus, on_level=dashboard.update_audio_level)
        listener.start()
        ...
        listener.stop()
    """

    def __init__(
        self,
        event_bus: EventBus,
        on_level: Callable[[float], None] | None = None,
    ):
        self.event_bus = event_bus
        self.on_level = on_level

        self.vad = SileroVAD()
        self.stt = SpeechToText()
        self.capture = AudioCapture(
            on_frame=self._on_audio_frame,
            on_level=self.on_level,
        )

        # Utterance state machine
        self._is_speaking = False
        self._silence_chunks = 0
        self._voice_chunks_count = 0
        self._speech_event_published = False
        self._utterance_buffer: list[np.ndarray] = []
        self._speech_start_time = 0.0

        # Pre-roll ring buffer: keeps the last 8 chunks (~256ms) of audio
        self._preroll_buffer: collections.deque = collections.deque(maxlen=8)

        # Background worker for transcription
        self._transcribe_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._running = False

        # Half-duplex mic ducking: ignore mic while ULTRON is speaking aloud
        self._ducked_until: float = 0.0
        self.event_bus.subscribe(EventTypes.SPEAKING_STARTED, self._on_speaking_started)
        self.event_bus.subscribe(EventTypes.SPEAKING_FINISHED, self._on_speaking_finished)

    def _on_speaking_started(self, event):
        """Duck microphone while ULTRON is speaking."""
        self._ducked_until = float("inf")
        if self._is_speaking:
            self._is_speaking = False
            self._utterance_buffer.clear()
            self._preroll_buffer.clear()

    def _on_speaking_finished(self, event):
        """Un-duck microphone after room reverb settles."""
        reverb_tail = getattr(config, "TTS_REVERB_TAIL_MS", 250) / 1000.0
        self._ducked_until = time.time() + reverb_tail

    def load(self) -> bool:
        """Initialize VAD and STT models."""
        print("[ULTRON Audio] Initializing audio models...")
        vad_ok = self.vad.load()
        stt_ok = self.stt.load()
        return vad_ok and stt_ok

    def start(self) -> bool:
        """Start background worker and audio stream."""
        if not self.vad.is_loaded or not self.stt.is_loaded:
            if not self.load():
                return False

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self._worker_thread.start()

        return self.capture.start()

    def stop(self):
        """Stop audio stream and worker thread."""
        self._running = False
        self.capture.stop()
        if self._worker_thread is not None:
            self._transcribe_queue.put(np.array([], dtype=np.float32))  # Poison pill
            self._worker_thread.join(timeout=2.0)
        print("[ULTRON Audio] Audio listener shut down.")

    def _on_audio_frame(self, frame: np.ndarray):
        """Callback invoked by AudioCapture for each 512-sample block (32ms)."""
        if not self._running:
            return

        # Half-duplex mic ducking: ignore mic while ULTRON is speaking + reverb tail
        if time.time() < self._ducked_until:
            if self._is_speaking:
                self._is_speaking = False
                self._utterance_buffer.clear()
                self._preroll_buffer.clear()
            return

        is_voice, prob = self.vad.is_speech(frame)

        if is_voice:
            self._silence_chunks = 0
            self._voice_chunks_count += 1

            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = time.time()
                self._speech_event_published = False
                # Prepend the pre-roll buffer to preserve the first syllable
                self._utterance_buffer = list(self._preroll_buffer)
                self._preroll_buffer.clear()

            self._utterance_buffer.append(frame)

            # Publish SPEECH_DETECTED only once sustained voice (>= 3 chunks) is established
            if not self._speech_event_published and self._voice_chunks_count >= 3:
                self._speech_event_published = True
                self.event_bus.publish(EventTypes.SPEECH_DETECTED, {
                    "probability": prob,
                    "timestamp": self._speech_start_time,
                })

            # Safety cutoff: if utterance exceeds maximum duration, finalize immediately
            elapsed = time.time() - self._speech_start_time
            if elapsed >= config.VAD_MAX_SPEECH_DURATION_S:
                self._finalize_utterance(reason="max duration reached")

        elif self._is_speaking:
            # Silence chunk during an ongoing speech utterance
            self._utterance_buffer.append(frame)
            self._silence_chunks += 1

            if self._silence_chunks >= config.VAD_SILENCE_TAIL_CHUNKS:
                self._finalize_utterance(reason="silence endpoint")

        else:
            # Idle silence: keep rolling the pre-roll buffer
            self._preroll_buffer.append(frame)

    def _finalize_utterance(self, reason: str = ""):
        """Finalize current speech segment and dispatch to transcription queue."""
        self._is_speaking = False
        self._silence_chunks = 0
        voice_count = self._voice_chunks_count
        self._voice_chunks_count = 0
        self._speech_event_published = False
        self.vad.reset_state()

        if not self._utterance_buffer:
            return

        full_utterance = np.concatenate(self._utterance_buffer)
        self._utterance_buffer = []

        # Filter 1: Discard transient noises (coughs, clicks, keypresses) with too few voice frames
        if voice_count < config.VAD_MIN_VOICE_CHUNKS:
            return

        # Filter 2: Discard segments that are too short overall
        min_samples = int(
            config.AUDIO_SAMPLE_RATE * (config.VAD_MIN_SPEECH_DURATION_MS / 1000.0)
        )
        if len(full_utterance) < min_samples:
            return

        # Filter 3: Discard faint background audio with near-zero peak amplitude
        peak = float(np.max(np.abs(full_utterance)))
        if peak < config.VAD_MIN_PEAK_AMPLITUDE:
            return

        duration_s = len(full_utterance) / config.AUDIO_SAMPLE_RATE
        print(f"[ULTRON Audio] Speech segment captured: {duration_s:.1f}s ({voice_count} voice frames, peak: {peak:.2f})")
        self._transcribe_queue.put(full_utterance)

    def _transcription_worker(self):
        """Background thread that consumes utterances and runs faster-whisper."""
        while self._running:
            try:
                audio_data = self._transcribe_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if len(audio_data) == 0:
                break  # Sentinel stop

            # Amplitude normalization
            peak = float(np.max(np.abs(audio_data)))
            if 0.001 < peak < 0.3:
                gain = min(10.0, 0.5 / peak)
                audio_data = audio_data * gain

            text, latency_ms = self.stt.transcribe(audio_data)
            text = text.strip()

            if text:
                print(f"[ULTRON STT] Valid English: \"{text}\" ({latency_ms:.0f}ms)")
                self.event_bus.publish(EventTypes.SPEECH_RECOGNIZED, {
                    "text": text,
                    "is_whisper": False,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                })
            else:
                print(f"[ULTRON STT] (Filtered out noise / non-English artifact in {latency_ms:.0f}ms)")
