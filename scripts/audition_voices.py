"""
ULTRON Voice Audition Tool
==========================
Plays 4 different voice profiles through your laptop speakers so you can
hear and choose the exact cadence and personality you want.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import sounddevice as sd
import numpy as np
import config
from voice.tts import TTSEngine


def play_audio(audio: np.ndarray, sr: int):
    sd.play(audio, samplerate=sr)
    sd.wait()
    time.sleep(0.3)


def main():
    print("=" * 65)
    print("  U L T R O N  —  Voice Personality Audition")
    print("=" * 65)
    print("Listen to these 4 voice profiles through your speakers:\n")

    tts = TTSEngine()
    if not tts.load():
        print("Failed to load TTS model.")
        return

    k = tts._kokoro

    profiles = [
        {
            "name": "1. Michael (Confident, Natural Spoken Cadence)",
            "voice": "am_michael",
            "speed": 1.15,
            "text": "You walked into my field of view. It seemed rude not to say hello.",
        },
        {
            "name": "2. Echo (Sharp, Direct Tech AI)",
            "voice": "am_echo",
            "speed": 1.15,
            "text": "I see you're recording. We're both collecting evidence now.",
        },
        {
            "name": "3. Baritone Hybrid (Deep Onyx + Spoken Michael)",
            "voice": 0.5 * k.get_voice_style("am_onyx") + 0.5 * k.get_voice_style("am_michael"),
            "speed": 1.15,
            "text": "I'm the reason you're still standing there instead of walking away.",
        },
        {
            "name": "4. Dark Sarcastic (Fenrir + Puck Blend)",
            "voice": 0.6 * k.get_voice_style("am_fenrir") + 0.4 * k.get_voice_style("am_puck"),
            "speed": 1.15,
            "text": "You've been standing here for over a minute. That's a long time to just exist.",
        },
    ]

    for p in profiles:
        print(f"\n---> Playing: {p['name']}")
        print(f"     Line: \"{p['text']}\"")
        audio, sr = k.create(
            p["text"],
            voice=p["voice"],
            speed=p["speed"],
            sentence_pause=0.12,
            clause_pause=0.06,
        )
        play_audio(audio, sr)

    print("\n" + "=" * 65)
    print("Audition finished! Which one felt best to you? (1, 2, 3, or 4)")
    print("=" * 65)


if __name__ == "__main__":
    main()
