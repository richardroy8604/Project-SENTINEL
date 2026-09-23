"""
ULTRON Test — Voice Synthesis & Playback Verification
======================================================
Tests Kokoro-82M ONNX speech synthesis with the 'am_onyx' voice
and verifies real-time sound playback through speakers.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.event_bus import EventBus, EventTypes
from voice.tts import TTSEngine
from voice.playback import VoicePlayback


def main():
    print("=" * 60)
    print("  ULTRON Voice Subsystem — Verification Test")
    print("=" * 60)
    print()

    # 1. Initialize Event Bus
    bus = EventBus()
    events_log = []

    def on_event(event):
        events_log.append((time.time(), event.event_type, event.data))
        if event.event_type == EventTypes.SPEAKING_STARTED:
            print(f"\n[EVENT] >>> SPEAKING STARTED (Mic ducked) -- \"{event.data.get('text')}\"")
        elif event.event_type == EventTypes.SPEAKING_FINISHED:
            print(f"[EVENT] <<< SPEAKING FINISHED (Mic un-ducked after reverb tail)\n")

    bus.subscribe(EventTypes.SPEAKING_STARTED, on_event)
    bus.subscribe(EventTypes.SPEAKING_FINISHED, on_event)

    # 2. Test TTS Engine in isolation
    print("[TEST 1] Testing TTSEngine standalone synthesis...")
    tts = TTSEngine()
    if not tts.load():
        print("FAIL: TTSEngine failed to load.")
        return 1

    sample_lines = [
        "Smile. You are on camera.",
        "I may have made the situation slightly more social.",
        "You walked into my field of view. It seemed rude not to say hello.",
    ]

    for line in sample_lines:
        audio, sr, lat = tts.synthesize(line)
        dur = len(audio) / sr
        print(f"  -> \"{line}\"")
        print(f"     Synthesized: {len(audio)} samples @ {sr}Hz ({dur:.2f}s) in {lat:.0f}ms")
        assert len(audio) > 0, "Synthesized audio is empty!"
        assert sr == 24000, f"Expected 24000Hz, got {sr}Hz"

    print("\n[TEST 1 PASSED] Synthesis verified.\n")

    # 3. Test VoicePlayback through speakers
    print("[TEST 2] Testing VoicePlayback with sounddevice output...")

    levels = []
    def on_level(lvl):
        levels.append(lvl)

    player = VoicePlayback(event_bus=bus, tts=tts, on_level=on_level)
    if not player.start():
        print("FAIL: VoicePlayback failed to start.")
        return 1

    test_line = "ULTRON voice online. Systems monitoring at full capacity."
    print(f"Speaking aloud through laptop speakers: \"{test_line}\"")
    player.speak(test_line)

    # Wait for playback to complete
    timeout = 10.0
    start_wait = time.time()
    while player.is_speaking or not any(e[1] == EventTypes.SPEAKING_STARTED for e in events_log):
        time.sleep(0.1)
        if time.time() - start_wait > timeout:
            break

    # Wait for finishing event
    while player.is_speaking:
        time.sleep(0.1)

    time.sleep(0.5)
    player.stop()

    # Verify events fired
    event_types = [e[1] for e in events_log]
    print("Events received:", set(event_types))
    assert EventTypes.SPEAKING_STARTED in event_types, "SPEAKING_STARTED was not published!"
    assert EventTypes.SPEAKING_FINISHED in event_types, "SPEAKING_FINISHED was not published!"
    assert len(levels) > 0, "No audio levels were captured for UI orb pulsation!"

    print("\n[TEST 2 PASSED] Sound playback, event orchestration, and audio levels verified.")
    print("=" * 60)
    print("  STAGE 6 VOICE OUTPUT VERIFICATION COMPLETE: ALL SYSTEMS NOMINAL")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
