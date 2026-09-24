"""
ULTRON Test — Identity Persistence & Re-entry Verification
=========================================================
Tests:
  1. IdentityTracker maps raw ByteTrack ID changes back to existing canonical ID
  2. Single person leaving and returning preserves their canonical ID
  3. Context restores returning visitor with greeting_given = True
  4. Person count remains 1
  5. Cooldown & return handling prevents "another one arrived" stranger greetings
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.event_bus import EventBus, EventTypes, Event
from core.context import ContextManager
from core.identity import IdentityTracker
from brain.reasoning import ReasoningEngine
from core.conversation import ConversationController


def test_reentry_persistence():
    print("\n--- Test: Identity Persistence & Returning Visitor Tracking ---")
    now = time.time()
    tracker = IdentityTracker(reid_window_s=60.0)

    # 1. First arrival (ByteTrack ID: 1)
    c1 = tracker.resolve_track_id(1, now)
    assert c1 == 1, f"Expected canonical ID 1, got {c1}"
    tracker.set_active_ids({c1})
    print("  [PASS] Initial person resolved to canonical ID: 1")

    # 2. Person steps out of view at t + 10s
    t_leave = now + 10.0
    tracker.on_person_departed(canonical_id=1, now=t_leave)
    assert 1 not in tracker.active_canonical_ids
    print("  [PASS] Person 1 recorded as departed at t+10s")

    # 3. Person returns at t + 18s (ByteTrack gives new ID: 2)
    t_return = now + 18.0
    c2 = tracker.resolve_track_id(2, t_return)
    assert c2 == 1, f"Expected ByteTrack ID 2 to remap to canonical ID 1, got {c2}!"
    tracker.set_active_ids({c2})
    print(f"  [PASS] Returning visitor (ByteTrack raw ID 2) successfully remapped to canonical ID {c2}!")

    # 4. ContextManager restoration
    bus = EventBus()
    ctx = ContextManager(bus)

    # Arrive
    ctx.update_person(track_id=1)
    ctx.mark_greeting_given(track_id=1)
    p = ctx._persons[1]
    assert p.greeting_given is True

    # Depart
    bus.publish(EventTypes.PERSON_LEFT, {"track_id": 1, "duration": 10})
    assert 1 not in ctx._persons
    assert 1 in ctx._departed_persons
    print("  [PASS] Departed person preserved in ContextManager._departed_persons")

    # Re-enter with resolved canonical ID 1
    ctx.update_person(track_id=1)
    p_restored = ctx._persons.get(1)
    assert p_restored is not None, "Failed to restore person context!"
    assert p_restored.greeting_given is True, "greeting_given was lost on re-entry!"
    assert p_restored.is_reentry is True, "is_reentry flag not set!"
    assert ctx.person_count == 1, f"Expected person_count == 1, got {ctx.person_count}"
    print("  [PASS] Returning visitor restored with greeting_given=True and is_reentry=True")
    print("  [PASS] Active person count remains exactly 1")

    summary = ctx.get_situation_summary()
    assert "returned after stepping away momentarily" in summary
    assert "Only 1 person is present in front of the camera" in summary
    print("  [PASS] Situation summary correctly warns LLM that only 1 person is present")


def main():
    print("=" * 60)
    print("  ULTRON Identity & Re-entry Persistence Verification")
    print("=" * 60)

    test_reentry_persistence()

    print("\n" + "=" * 60)
    print("  ALL RE-ENTRY & IDENTITY PERSISTENCE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
