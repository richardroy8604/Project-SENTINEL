"""
ULTRON Core — Context Manager
===============================
Short-term memory for ULTRON. Tracks everything the AI needs to know
about the current situation:

  - Who is present (IDs, dwell times)
  - Conversation state (greeting given, last speech)
  - Security state
  - Recent events

This context is assembled into a structured prompt for the LLM
so ULTRON always knows what's happening right now.
"""

import time
import threading
from dataclasses import dataclass, field

from core.event_bus import EventBus, EventTypes, Event
import config


@dataclass
class PersonContext:
    """Tracking context for a single person."""

    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    greeting_given: bool = False

    @property
    def dwell_time(self) -> float:
        """How long this person has been present (seconds)."""
        return self.last_seen - self.first_seen


class ContextManager:
    """
    ULTRON's short-term memory.

    Maintains awareness of:
      - Current persons and their dwell times
      - Whether greetings have been given
      - Conversation history
      - Current security state
      - Recent events summary

    This is the single source of truth that gets fed to the LLM.
    """

    def __init__(self, event_bus: EventBus):
        self._bus = event_bus
        self._lock = threading.Lock()

        # Person tracking
        self._persons: dict[int, PersonContext] = {}

        # Conversation
        self._conversation_history: list[dict[str, str]] = []
        self._max_conversation = 20  # Keep last N exchanges

        # Security
        self._security_state = config.SecurityState.MONITORING
        self._security_reason = ""

        # Audio
        self._last_speech_text: str = ""
        self._last_speech_time: float = 0.0
        self._is_whisper: bool = False

        # Subscribe to events
        self._bus.subscribe(EventTypes.PERSON_ENTERED, self._on_person_entered)
        self._bus.subscribe(EventTypes.PERSON_LEFT, self._on_person_left)
        self._bus.subscribe(EventTypes.STATE_CHANGED, self._on_state_changed)
        self._bus.subscribe(EventTypes.SPEECH_RECOGNIZED, self._on_speech)
        self._bus.subscribe(EventTypes.RESPONSE_GENERATED, self._on_response)

    # ── Person Tracking ─────────────────────────────────────────────

    def update_person(self, track_id: int):
        """Update the last-seen time for a tracked person."""
        with self._lock:
            if track_id in self._persons:
                self._persons[track_id].last_seen = time.time()
            else:
                self._persons[track_id] = PersonContext(track_id=track_id)

    def mark_greeting_given(self, track_id: int):
        """Mark that ULTRON has greeted this person."""
        with self._lock:
            if track_id in self._persons:
                self._persons[track_id].greeting_given = True

    def get_ungreeted_persons(self) -> list[PersonContext]:
        """Get persons who haven't been greeted yet."""
        with self._lock:
            return [
                p for p in self._persons.values() if not p.greeting_given
            ]

    @property
    def person_count(self) -> int:
        with self._lock:
            return len(self._persons)

    @property
    def persons(self) -> list[PersonContext]:
        with self._lock:
            return list(self._persons.values())

    # ── Conversation ────────────────────────────────────────────────

    def add_conversation(self, role: str, text: str):
        """Add a conversation entry (role: 'human' or 'ultron')."""
        with self._lock:
            self._conversation_history.append({
                "role": role,
                "text": text,
                "time": time.time(),
            })
            if len(self._conversation_history) > self._max_conversation:
                self._conversation_history = self._conversation_history[
                    -self._max_conversation:
                ]

    @property
    def conversation_history(self) -> list[dict]:
        with self._lock:
            return list(self._conversation_history)

    # ── Context Assembly (for LLM) ──────────────────────────────────

    def get_situation_summary(self) -> str:
        """
        Assemble a structured text summary of the current situation.
        This gets injected into the LLM prompt as context.
        """
        with self._lock:
            lines = ["[CURRENT SITUATION]"]
            lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Security state: {self._security_state}")
            if self._security_reason:
                lines.append(f"State reason: {self._security_reason}")

            lines.append(f"Persons detected: {len(self._persons)}")
            for p in self._persons.values():
                dwell = int(p.dwell_time)
                greeted = "YES" if p.greeting_given else "NO"
                lines.append(
                    f"  Person ID:{p.track_id} — present {dwell}s, greeted: {greeted}"
                )

            if self._last_speech_text:
                lines.append(f"Last person speech: \"{self._last_speech_text}\"")
                whisper_tag = " (WHISPERED)" if self._is_whisper else ""
                lines.append(f"Speech type: normal{whisper_tag}")
                ago = int(time.time() - self._last_speech_time)
                lines.append(f"Speech was {ago}s ago")

            # Recent conversation
            recent = self._conversation_history[-4:]  # Last 4 exchanges
            if recent:
                lines.append("Recent conversation:")
                for entry in recent:
                    role = "PERSON" if entry["role"] == "human" else "ULTRON"
                    lines.append(f"  {role}: \"{entry['text']}\"")

            return "\n".join(lines)

    # ── Event Handlers ──────────────────────────────────────────────

    def _on_person_entered(self, event: Event):
        track_id = event.data.get("track_id", -1)
        if track_id >= 0:
            with self._lock:
                if track_id not in self._persons:
                    self._persons[track_id] = PersonContext(track_id=track_id)

    def _on_person_left(self, event: Event):
        track_id = event.data.get("track_id", -1)
        with self._lock:
            self._persons.pop(track_id, None)

    def _on_state_changed(self, event: Event):
        with self._lock:
            self._security_state = event.data.get("new_state", "UNKNOWN")
            self._security_reason = event.data.get("reason", "")

    def _on_speech(self, event: Event):
        with self._lock:
            self._last_speech_text = event.data.get("text", "")
            self._last_speech_time = time.time()
            self._is_whisper = event.data.get("is_whisper", False)
            self._conversation_history.append({
                "role": "human",
                "text": self._last_speech_text,
                "time": time.time(),
            })

    def _on_response(self, event: Event):
        text = event.data.get("text", "")
        if text:
            with self._lock:
                self._conversation_history.append({
                    "role": "ultron",
                    "text": text,
                    "time": time.time(),
                })
