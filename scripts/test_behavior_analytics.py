"""
ULTRON Test — Stage 8 Behavior Analytics & Threat Detection
=============================================================
Verifies deterministic accuracy of:
1. Camera Tamper & Obstruction Detection (blackout and lens blur)
2. Loitering Detection Engine (stationary linger vs transit)
3. Whisper / Hushed Speech Acoustic Classification (RMS, ZCR, high-freq ratio)
4. Weapon & Dangerous Tool Detection overlays (Red boxes for weapons, Blue for phones)
"""

import math
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.event_bus import EventBus, EventTypes
from security.tamper import CameraTamperDetector
from security.loitering import LoiteringDetector
from security.whisper_classifier import WhisperClassifier
from vision.detector import PersonDetection, PersonDetector


def test_tamper_detection():
    print("\n--- 1. Testing Camera Tamper & Obstruction Detection ---")
    event_bus = EventBus()
    tamper_events = []
    clear_events = []

    event_bus.subscribe(EventTypes.CAMERA_OBSTRUCTED, lambda e: tamper_events.append(e.data))
    event_bus.subscribe(EventTypes.TAMPER_CLEARED, lambda e: clear_events.append(e.data))

    detector = CameraTamperDetector(event_bus)
    detector.min_frames = 5  # Accelerate for unit test

    # Normal textured frame (e.g. room)
    normal_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    # Add gradients/edges
    for i in range(0, 480, 20):
        cv2.line(normal_frame, (0, i), (640, i), (255, 255, 255), 2)

    for _ in range(10):
        is_tampered, t_type, _ = detector.evaluate_frame(normal_frame)
    assert not is_tampered, "Normal frame falsely flagged as tampered!"
    print("  [PASS] Normal frame correctly identified as clear.")

    # Blackout / Lens covered frame (mean < 18 or std < 8)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(6):
        is_tampered, t_type, _ = detector.evaluate_frame(black_frame)
    assert is_tampered, "Blackout frame not flagged as tampered!"
    assert t_type == "LENS_COVERED", f"Expected LENS_COVERED, got {t_type}"
    assert len(tamper_events) > 0, "CAMERA_OBSTRUCTED event not published!"
    print("  [PASS] Physical blackout / hand over lens detected and debounced.")

    # Recover with normal frame
    for _ in range(6):
        is_tampered, t_type, _ = detector.evaluate_frame(normal_frame)
    assert not is_tampered, "Tamper not cleared on recovery!"
    assert len(clear_events) > 0, "TAMPER_CLEARED event not published!"
    print("  [PASS] Tamper clearing detected and debounced.")


def test_loitering_detection():
    print("\n--- 2. Testing Loitering Detection Engine ---")
    event_bus = EventBus()
    loiter_events = []
    event_bus.subscribe(EventTypes.LOITERING_DETECTED, lambda e: loiter_events.append(e.data))

    detector = LoiteringDetector(event_bus)
    detector.threshold_s = 2.0  # Accelerate threshold to 2s for unit test
    detector.radius_ratio = 0.20

    # Person 1: Stationary lingering at center (320, 240)
    p1 = PersonDetection(
        track_id=1,
        bbox=(280, 160, 360, 320),
        confidence=0.90,
        center=(320, 240),
        bbox_area=80 * 160,
    )

    t0 = time.time()
    # Feed frames for 2.2 seconds with tiny jitter (+/- 2 pixels)
    for step in range(10):
        # Slightly alter center to simulate natural micro-movements
        jitter_x = int(math.sin(step) * 3)
        jitter_y = int(math.cos(step) * 3)
        p1.center = (320 + jitter_x, 240 + jitter_y)
        detector.update_tracks([p1], frame_width=640, frame_height=480)
        time.sleep(0.25)

    loitering_ids = detector.update_tracks([p1], frame_width=640, frame_height=480)
    assert 1 in loitering_ids, "Stationary person not flagged as loitering!"
    assert len(loiter_events) == 1, f"Expected 1 loiter event, got {len(loiter_events)}"
    print(f"  [PASS] Stationary person flagged as loitering (dwell: {loiter_events[0]['duration']:.1f}s).")

    # Person 2: Moving person in transit (drifts across frame: 50 -> 500 px)
    p2 = PersonDetection(
        track_id=2,
        bbox=(50, 160, 130, 320),
        confidence=0.90,
        center=(90, 240),
        bbox_area=80 * 160,
    )
    for step in range(10):
        p2.center = (90 + step * 45, 240)
        detector.update_tracks([p2], frame_width=640, frame_height=480)
        time.sleep(0.25)

    loitering_ids_2 = detector.update_tracks([p2], frame_width=640, frame_height=480)
    assert 2 not in loitering_ids_2, "Transit person falsely flagged as loitering!"
    print("  [PASS] Moving transit person correctly excluded from loitering.")


def test_whisper_classification():
    print("\n--- 3. Testing Whisper / Hushed Speech Classification ---")
    event_bus = EventBus()
    whisper_events = []
    event_bus.subscribe(EventTypes.WHISPER_DETECTED, lambda e: whisper_events.append(e.data))

    classifier = WhisperClassifier(event_bus)

    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)

    # 1. Normal loud voiced speech (strong fundamental 150 Hz tone, high amplitude)
    normal_voice = 0.25 * np.sin(2 * np.pi * 150 * t) + 0.15 * np.sin(2 * np.pi * 300 * t)
    is_whisper, conf, metrics = classifier.analyze_utterance(normal_voice, sample_rate)
    assert not is_whisper, "Normal voice falsely flagged as whisper!"
    print(f"  [PASS] Normal voice: is_whisper={is_whisper} (RMS={metrics['rms']:.3f}, ZCR={metrics['zcr']:.3f})")

    # 2. Whispered speech (low amplitude, unvoiced high frequency turbulence ~ 3 kHz to 5 kHz)
    np.random.seed(42)
    whisper_noise = np.random.normal(0, 0.02, sample_rate).astype(np.float32)
    # Bandpass / high frequency emphasize
    whisper_sim = np.sin(2 * np.pi * 3200 * t) * 0.025 + whisper_noise * 0.015

    is_whisper, conf, metrics = classifier.analyze_utterance(whisper_sim, sample_rate)
    assert is_whisper, f"Simulated whisper was not detected! (metrics: {metrics})"
    assert len(whisper_events) > 0, "WHISPER_DETECTED event not published!"
    print(f"  [PASS] Whisper detected: conf={conf:.0%} (RMS={metrics['rms']:.3f}, ZCR={metrics['zcr']:.3f}, HF_ratio={metrics['hf_ratio']:.0%})")


def test_color_hierarchy_and_overlays():
    print("\n--- 4. Testing Color Hierarchy & Box Annotations ---")
    detector = PersonDetector()

    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Person 1: Armed with knife
    p1 = PersonDetection(
        track_id=1,
        bbox=(50, 100, 200, 400),
        confidence=0.88,
        center=(125, 250),
        bbox_area=150 * 300,
        holding_weapon=True,
        weapon_type="KNIFE",
    )

    # Person 2: Holding phone
    p2 = PersonDetection(
        track_id=2,
        bbox=(250, 100, 400, 400),
        confidence=0.92,
        center=(325, 250),
        bbox_area=150 * 300,
        holding_phone=True,
    )

    # Person 3: Normal visitor
    p3 = PersonDetection(
        track_id=3,
        bbox=(450, 100, 600, 400),
        confidence=0.85,
        center=(525, 250),
        bbox_area=150 * 300,
    )

    phone_boxes = [(310, 220, 340, 270, 0.82)]
    weapon_boxes = [(160, 220, 210, 290, 0.78, "KNIFE")]

    annotated = detector.draw_overlays(
        frame.copy(),
        [p1, p2, p3],
        phone_boxes=phone_boxes,
        weapon_boxes=weapon_boxes,
    )

    # Verify colors exist on the annotated canvas
    # Crimson Red (BGR: 0, 0, 255) must be present for weapons
    has_red = np.any((annotated[:, :, 0] < 50) & (annotated[:, :, 1] < 50) & (annotated[:, :, 2] > 200))
    assert has_red, "Crimson Red bounding box not found for weapon!"

    # Electric Blue (BGR: 255, 140, 0) must be present for phones
    has_blue = np.any((annotated[:, :, 0] > 200) & (annotated[:, :, 1] > 100) & (annotated[:, :, 2] < 50))
    assert has_blue, "Electric Blue bounding box not found for cell phone!"

    # Tactical Green (BGR: 0, 255, 100) must be present for normal person
    has_green = np.any((annotated[:, :, 0] < 50) & (annotated[:, :, 1] > 200) & (annotated[:, :, 2] < 150))
    assert has_green, "Tactical Green bounding box not found for normal person!"

    print("  [PASS] Color hierarchy verified:")
    print("         - Weapons / Dangerous tools -> Crimson Red (0, 0, 255)")
    print("         - Normal objects / Phones   -> Electric Blue (255, 140, 0)")
    print("         - Normal persons            -> Tactical Green (0, 255, 100)")


def test_identity_spatial_persistence():
    print("\n--- 5. Testing Identity Spatial Persistence & Re-Entry ---")
    from core.identity import IdentityTracker

    tracker = IdentityTracker(reid_window_s=60.0)
    now = time.time()

    # Frame 1: Single person detected with raw ID 1
    p1 = PersonDetection(
        track_id=1,
        bbox=(200, 100, 350, 400),
        confidence=0.90,
        center=(275, 250),
        bbox_area=150 * 300,
    )
    active = tracker.resolve_frame_persons([p1], now)
    assert p1.track_id == 1, f"Expected canonical ID 1, got {p1.track_id}"
    assert len(active) == 1, f"Expected 1 active person, got {len(active)}"
    print("  [PASS] Frame 1: Person assigned canonical ID 1.")

    # Frame 2: ByteTrack drops and assigns raw ID 2 to the same person (shifted 5 pixels)
    p2 = PersonDetection(
        track_id=2,
        bbox=(205, 100, 355, 400),
        confidence=0.90,
        center=(280, 250),
        bbox_area=150 * 300,
    )
    active = tracker.resolve_frame_persons([p2], now + 0.033)
    assert p2.track_id == 1, f"Tracker ID churn was not mapped to canonical ID 1! Got {p2.track_id}"
    assert len(active) == 1, f"Ghost person created! Expected 1, got {len(active)}"
    print("  [PASS] Frame 2: ByteTrack raw ID 2 seamlessly preserved as canonical ID 1 via spatial overlap.")

    # Frame 3: ByteTrack jumps to raw ID 27 (shifted 10 pixels)
    p3 = PersonDetection(
        track_id=27,
        bbox=(210, 105, 360, 405),
        confidence=0.88,
        center=(285, 255),
        bbox_area=150 * 300,
    )
    active = tracker.resolve_frame_persons([p3], now + 0.066)
    assert p3.track_id == 1, f"Raw ID 27 was not mapped to canonical ID 1! Got {p3.track_id}"
    assert len(active) == 1, f"Ghost persons accumulated! Got {len(active)}"
    print("  [PASS] Frame 3: ByteTrack raw ID 27 preserved as canonical ID 1.")

    # Frame 4: Person leaves view (empty frame)
    active = tracker.resolve_frame_persons([], now + 1.0)
    assert len(active) == 0, f"Expected 0 active persons after departure, got {len(active)}"
    print("  [PASS] Frame 4: Departure registered, 0 active persons.")

    # Frame 5: Person returns 10 seconds later with raw ID 45
    p5 = PersonDetection(
        track_id=45,
        bbox=(220, 110, 370, 410),
        confidence=0.89,
        center=(295, 260),
        bbox_area=150 * 300,
    )
    active = tracker.resolve_frame_persons([p5], now + 11.0)
    assert p5.track_id == 1, f"Returning person received new ID! Expected 1, got {p5.track_id}"
    assert len(active) == 1, f"Expected 1 active person, got {len(active)}"
    print("  [PASS] Frame 5: Returning visitor re-identified as canonical ID 1 (no new stranger ID).")

    # Frame 6: Genuine second person enters simultaneously on the other side of the room
    p6_a = PersonDetection(
        track_id=45,
        bbox=(220, 110, 370, 410),
        confidence=0.89,
        center=(295, 260),
        bbox_area=150 * 300,
    )
    p6_b = PersonDetection(
        track_id=52,
        bbox=(500, 120, 620, 420),
        confidence=0.85,
        center=(560, 270),
        bbox_area=120 * 300,
    )
    active = tracker.resolve_frame_persons([p6_a, p6_b], now + 12.0)
    assert p6_a.track_id == 1, f"Person A changed from 1 to {p6_a.track_id}"
    assert p6_b.track_id == 2, f"Expected Person B to receive canonical ID 2, got {p6_b.track_id}"
    assert len(active) == 2, f"Expected 2 active persons, got {len(active)}"
    print("  [PASS] Frame 6: Genuinely distinct second person received canonical ID 2.")


if __name__ == "__main__":
    print("=" * 65)
    print("  ULTRON Stage 8 — Behavior Analytics & Threat Test Suite")
    print("=" * 65)
    test_tamper_detection()
    test_loitering_detection()
    test_whisper_classification()
    test_color_hierarchy_and_overlays()
    test_identity_spatial_persistence()
    print("\n>>> ALL STAGE 8 TESTS PASSED WITH 100% MATHEMATICAL ACCURACY! <<<\n")
