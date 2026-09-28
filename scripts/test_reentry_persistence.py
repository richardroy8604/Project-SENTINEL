"""
ULTRON Test — Identity Persistence & Re-entry Verification
=========================================================
Tests:
  1. IdentityTracker maps raw ByteTrack ID changes back to existing canonical ID
  2. Single person leaving and returning preserves their canonical ID
  3. Momentary absence (< 25s) does NOT trigger re-entry greeting (is_reentry == False)
  4. Genuine departure (>= 25s) sets is_reentry == True and preserves greeting_given
  5. Preemption verification: User speech preempts autonomous generation
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.event_bus import EventBus, EventTypes, Event
from core.context import ContextManager
from core.identity import IdentityTracker
from vision.detector import PersonDetection


def test_reentry_persistence():
    print("\n--- Test 1: IdentityTracker Spatial & Re-Entry Persistence ---")
    now = time.time()
    tracker = IdentityTracker(reid_window_s=60.0)

    # 1. First arrival (ByteTrack ID: 1)
    p1 = PersonDetection(
        track_id=1,
        bbox=(100, 100, 200, 300),
        confidence=0.85,
        center=(150, 200),
        bbox_area=100 * 200,
    )
    c1_set = tracker.resolve_frame_persons([p1], now)
    assert 1 in c1_set, f"Expected canonical ID 1, got {c1_set}"
    assert p1.track_id == 1
    print("  [PASS] Initial person resolved to canonical ID: 1")

    # 2. Raw ID jitter (ByteTrack raw ID jumps 1 -> 2 at similar position)
    p2 = PersonDetection(
        track_id=2,
        bbox=(105, 102, 205, 305),
        confidence=0.82,
        center=(155, 203),
        bbox_area=100 * 203,
    )
    c2_set = tracker.resolve_frame_persons([p2], now + 0.1)
    assert 1 in c2_set, f"Expected raw ID 2 to remap to canonical ID 1, got {c2_set}"
    assert p2.track_id == 1
    print("  [PASS] Spatial persistence preserved canonical ID 1 across raw tracker jitter")

    # 3. Person steps out of view for 5 seconds (momentary absence)
    tracker.resolve_frame_persons([], now + 5.0)

    # 4. ContextManager: Test momentary departure (< 25.0s)
    bus = EventBus()
    ctx = ContextManager(bus)

    ctx.update_person(track_id=1)
    ctx.mark_greeting_given(track_id=1)
    assert ctx._persons[1].greeting_given is True

    # Person leaves
    bus.publish(EventTypes.PERSON_LEFT, {"track_id": 1})
    assert 1 not in ctx._persons
    assert 1 in ctx._departed_persons
    print("  [PASS] Departed person preserved in ContextManager._departed_persons")

    # Person returns 3 seconds later (flicker / momentary absence)
    ctx.update_person(track_id=1)
    p_short = ctx._persons.get(1)
    assert p_short is not None
    assert p_short.greeting_given is True, "greeting_given must be preserved!"
    assert p_short.is_reentry is False, "Momentary departure (<25s) must NOT set is_reentry=True!"
    print("  [PASS] Momentary absence (<25s) correctly prevented false re-entry trigger (is_reentry=False)")

    # 5. ContextManager: Test genuine departure (>= 25.0s)
    # Simulate leaving and staying gone for 30s
    p_short.departed_at = time.time() - 30.0
    ctx._departed_persons[1] = p_short
    del ctx._persons[1]

    # Person re-enters after 30 seconds
    ctx.update_person(track_id=1)
    p_genuine = ctx._persons.get(1)
    assert p_genuine is not None
    assert p_genuine.greeting_given is True, "greeting_given must still be preserved!"
    assert p_genuine.is_reentry is True, "Genuine departure (>=25s) MUST set is_reentry=True!"
    print("  [PASS] Genuine departure (>=25s) correctly flagged as is_reentry=True")

    summary = ctx.get_situation_summary()
    assert "returned after stepping away momentarily" in summary
    assert "Only 1 person is present in front of the camera" in summary
    print("  [PASS] Situation summary correctly formats re-entry status for LLM context")


def test_generation_preemption():
    print("\n--- Test 2: User Speech Preemption of Autonomous Workers ---")
    from brain.reasoning import ReasoningEngine

    bus = EventBus()
    ctx = ContextManager(bus)
    engine = ReasoningEngine(bus, ctx)

    # Initial state
    assert engine._generation_id == 0
    assert engine._busy is False

    # Autonomous greeting starts
    started = engine.generate_autonomous_greeting(track_id=1)
    assert started is True
    assert engine._generation_id == 1
    assert engine.is_busy is True

    # User speech arrives while autonomous greeting is in-flight
    bus.publish(EventTypes.SPEECH_RECOGNIZED, {"text": "Who are you?"})
    time.sleep(0.1)

    # Generation ID must increment, preempting the autonomous worker
    assert engine._generation_id >= 2
    assert len(ctx.conversation_history) > 0
    assert ctx.conversation_history[-1]["role"] == "ultron"
    print("  [PASS] User speech successfully bumped generation_id and preempted autonomous pipeline")


def main():
    print("=" * 60)
    print("  ULTRON Identity & Re-entry Persistence Verification")
    print("=" * 60)

    test_reentry_persistence()
    test_generation_preemption()

    print("\n" + "=" * 60)
    print("  ALL RE-ENTRY & PREEMPTION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
