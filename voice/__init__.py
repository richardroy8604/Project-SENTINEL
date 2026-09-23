"""
ULTRON Voice Subsystem (Stage 6)
================================
Provides text-to-speech synthesis (Kokoro TTS ONNX) and real-time audio playback
with half-duplex mic ducking and audio level telemetry.
"""

from voice.tts import TTSEngine
from voice.playback import VoicePlayback

__all__ = ["TTSEngine", "VoicePlayback"]
