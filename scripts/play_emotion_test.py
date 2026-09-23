"""
ULTRON Audio Expressiveness Test
================================
Plays 3 options through your laptop speakers:
1. Kokoro Local (with new Prosodic Phrase Pausing)
2. Edge-TTS Christopher (Deep, emotional, human cadence)
3. Edge-TTS Guy (Casual, conversational, natural swagger)
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import sounddevice as sd
import soundfile as sf
import numpy as np


def play_file(filename: str, label: str):
    print(f"\n---> Playing: {label}")
    data, sr = sf.read(filename)
    sd.play(data, samplerate=sr)
    sd.wait()
    time.sleep(0.4)


def main():
    print("=" * 65)
    print("  U L T R O N  —  Emotional Cadence & Delivery Test")
    print("=" * 65)
    print("Line: \"I'm here... and you're standing on my doorstep.\"\n")

    # 1. Kokoro with dramatic phrase pause
    play_file("test_dramatic_pause.wav", "1. Kokoro Local (with Prosodic Phrase Pauses)")

    # 2. Edge-TTS Christopher
    play_file("test_edge_christopher.mp3", "2. Edge-TTS Christopher (Deep, emotional, human pauses)")

    # 3. Edge-TTS Guy
    play_file("test_edge_guy.mp3", "3. Edge-TTS Guy (Casual, conversational swagger)")

    print("\n" + "=" * 65)
    print("Which delivery has the emotion and conversational tone you want?")
    print("=" * 65)


if __name__ == "__main__":
    main()
