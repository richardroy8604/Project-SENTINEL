"""
ULTRON Security — Whisper & Hushed Speech Acoustic Classifier
=============================================================
Deterministic acoustic feature extraction on 16 kHz audio waveforms.

Whispered or conspiratorial speech exhibits unique physical acoustic properties:
  1. Low RMS Energy: Absence of strong vocal cord vibration.
  2. High Zero-Crossing Rate (ZCR): High-frequency unvoiced turbulence / fricatives.
  3. High-Frequency Spectral Bias: Absence of low-frequency glottal pitch harmonics;
     energy is concentrated in the 2 kHz – 6 kHz band.

When a speech segment matches these acoustic characteristics, it emits
WHISPER_DETECTED on the EventBus.
"""

import numpy as np

import config
from core.event_bus import EventBus, EventTypes


class WhisperClassifier:
    """
    Acoustic analyzer classifying normal vs whispered speech.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self.max_rms = getattr(config, "WHISPER_MAX_RMS", 0.045)
        self.min_zcr = getattr(config, "WHISPER_MIN_ZCR", 0.12)

    def analyze_utterance(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[bool, float, dict]:
        """
        Analyze an audio waveform segment (float32 array).

        Returns:
            (is_whisper, confidence, metrics)
        """
        if audio is None or len(audio) < 512:
            return False, 0.0, {}

        # 1. Compute RMS energy
        rms = float(np.sqrt(np.mean(audio**2)))

        # 2. Compute Zero-Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2.0
        zcr = float(zero_crossings / len(audio))

        # 3. High-frequency spectral energy ratio (above 2 kHz)
        fft_vals = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)

        total_energy = np.sum(fft_vals**2) + 1e-9
        high_freq_mask = freqs >= 2000.0
        high_freq_energy = np.sum(fft_vals[high_freq_mask]**2)
        hf_ratio = float(high_freq_energy / total_energy)

        metrics = {
            "rms": rms,
            "zcr": zcr,
            "hf_ratio": hf_ratio,
        }

        # Classification rule
        # A whisper has low RMS, elevated ZCR, and significant high-frequency acoustic friction
        is_whisper = False
        confidence = 0.0

        if rms > 0.005 and rms <= self.max_rms:
            score = 0.0
            if zcr >= self.min_zcr:
                score += 0.5
            if hf_ratio >= 0.40:
                score += 0.5

            if score >= 0.5:
                is_whisper = True
                confidence = float(score * (1.0 - (rms / self.max_rms) * 0.3))

        if is_whisper:
            self.event_bus.publish(EventTypes.WHISPER_DETECTED, {
                "confidence": confidence,
                "metrics": metrics,
            })

        return is_whisper, confidence, metrics
