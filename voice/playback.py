"""
ULTRON Voice — Playback Engine & Audio Dispatcher
==================================================
Manages queued speech synthesis and real-time audio playback using sounddevice.
Emits SPEAKING_STARTED / SPEAKING_FINISHED events for half-duplex mic ducking,
and streams audio playback energy for the animated dashboard AI orb.
"""

import math
import queue
import threading
import time
from typing import Callable
import numpy as np
import sounddevice as sd

import config
from core.event_bus import EventBus, EventTypes
from voice.tts import TTSEngine


class VoicePlayback:
    """
    Asynchronous voice playback system with event coordination.

    Usage:
        player = VoicePlayback(event_bus, tts_engine, on_level=dashboard.update_audio_level)
        player.start()
        player.speak("Smile. You're on camera.")
        ...
        player.stop()
    """

    def __init__(
        self,
        event_bus: EventBus,
        tts: TTSEngine | None = None,
        on_level: Callable[[float], None] | None = None,
    ):
        self.event_bus = event_bus
        self.tts = tts or TTSEngine()
        self.on_level = on_level

        self._queue: queue.Queue[str] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._is_speaking = False
        self._interrupted = threading.Event()

    @property
    def is_speaking(self) -> bool:
        """True if ULTRON is actively speaking audio through speakers."""
        return self._is_speaking

    def start(self) -> bool:
        """Start the background playback worker."""
        if self._running:
            return True

        if not self.tts.is_loaded:
            if not self.tts.load():
                print("[ULTRON Playback] Warning: TTS model failed to load. Voice output disabled.")
                return False

        self._running = True
        self._interrupted.clear()
        self._worker_thread = threading.Thread(
            target=self._playback_worker,
            name="UltronVoicePlayback",
            daemon=True,
        )
        self._worker_thread.start()
        print("[ULTRON Playback] Voice output engine started.")
        return True

    def stop(self):
        """Stop playback and terminate the worker thread."""
        self._running = False
        self.interrupt()
        self._queue.put("")  # Wake up worker queue
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        print("[ULTRON Playback] Voice playback engine stopped.")

    def speak(self, text: str, priority: bool = False):
        """
        Queue text for speech output.

        Args:
            text: Text to synthesize and speak
            priority: If True, clears any pending speech before queueing this message
        """
        if not text or not config.TTS_ENABLED:
            return

        if priority:
            self.interrupt()

        self._queue.put(text.strip())

    def interrupt(self):
        """Interrupt any current speech and clear pending speech queue."""
        self._interrupted.set()
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _playback_worker(self):
        """Background worker consuming speech text, synthesizing, and playing audio."""
        while self._running:
            try:
                text = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if not self._running or not text:
                continue

            self._interrupted.clear()

            # 1. Synthesize audio via Kokoro ONNX
            try:
                samples, sample_rate, synth_lat = self.tts.synthesize(text)
            except Exception as e:
                print(f"[ULTRON Playback] Synthesis error: {e}")
                continue

            if len(samples) == 0:
                continue

            duration_s = len(samples) / sample_rate
            print(f"[ULTRON Voice] Speaking ({synth_lat:.0f}ms synth, {duration_s:.1f}s audio): \"{text}\"")

            # 2. Notify system that ULTRON started speaking (mic will duck)
            self._is_speaking = True
            self.event_bus.publish(EventTypes.SPEAKING_STARTED, {
                "text": text,
                "duration": duration_s,
                "timestamp": time.time(),
            })

            # 3. Stream audio in chunks to measure live RMS level for the UI orb
            chunk_size = 1024
            try:
                with sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=chunk_size,
                ) as stream:
                    for i in range(0, len(samples), chunk_size):
                        if not self._running or self._interrupted.is_set():
                            print("[ULTRON Voice] Playback interrupted.")
                            break

                        chunk = samples[i:i + chunk_size]
                        # Pad last chunk if needed
                        if len(chunk) < chunk_size:
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                        # Compute RMS energy for orb reactivity
                        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
                        if self.on_level is not None:
                            try:
                                self.on_level(rms)
                            except Exception:
                                pass

                        self.event_bus.publish(EventTypes.AUDIO_PLAYBACK_LEVEL, {
                            "level": rms,
                            "timestamp": time.time(),
                        })

                        stream.write(chunk)

            except Exception as e:
                print(f"[ULTRON Playback] Sounddevice output error: {e}")

            finally:
                # Reset visual level
                if self.on_level is not None:
                    try:
                        self.on_level(0.0)
                    except Exception:
                        pass

                # 4. Wait for acoustic room reverb tail before unmuting mic
                reverb_tail = config.TTS_REVERB_TAIL_MS / 1000.0
                if reverb_tail > 0:
                    time.sleep(reverb_tail)

                self._is_speaking = False
                self.event_bus.publish(EventTypes.SPEAKING_FINISHED, {
                    "text": text,
                    "timestamp": time.time(),
                })
