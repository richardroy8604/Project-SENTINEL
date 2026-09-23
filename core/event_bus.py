"""
ULTRON Core — Event Bus
========================
Thread-safe publish/subscribe event system.

Modules publish events (PERSON_ENTERED, SPEECH_RECOGNIZED, etc.)
and other modules subscribe to handle them. This decouples all
components — vision doesn't need to know about the brain, etc.

Usage:
    bus = EventBus()
    bus.subscribe("PERSON_ENTERED", my_handler)
    bus.publish("PERSON_ENTERED", {"track_id": 1, "zone": "entrance"})
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import defaultdict


@dataclass
class Event:
    """A single event with type, data, and timestamp."""

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return f"Event({self.event_type}, {self.data})"


# Standard event type constants
class EventTypes:
    """All event types used in ULTRON."""

    # Vision events
    PERSON_ENTERED = "PERSON_ENTERED"
    PERSON_LEFT = "PERSON_LEFT"
    PERSON_TRACKING = "PERSON_TRACKING"
    CAMERA_OBSTRUCTED = "CAMERA_OBSTRUCTED"

    # Audio events
    SPEECH_DETECTED = "SPEECH_DETECTED"
    SPEECH_RECOGNIZED = "SPEECH_RECOGNIZED"
    WHISPER_DETECTED = "WHISPER_DETECTED"

    # Security events
    STATE_CHANGED = "STATE_CHANGED"
    LOITERING_DETECTED = "LOITERING_DETECTED"

    # Brain/Voice events
    RESPONSE_GENERATED = "RESPONSE_GENERATED"
    TTS_STARTED = "TTS_STARTED"
    TTS_FINISHED = "TTS_FINISHED"
    SPEAKING_STARTED = "TTS_STARTED"
    SPEAKING_FINISHED = "TTS_FINISHED"
    AUDIO_PLAYBACK_LEVEL = "AUDIO_PLAYBACK_LEVEL"


class EventBus:
    """
    Thread-safe publish/subscribe event bus.

    Any module can publish events, and any module can subscribe
    to specific event types. Handlers are called synchronously
    on the publisher's thread, so keep them lightweight.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._max_history = 200  # Keep last N events

    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """
        Subscribe a handler to an event type.

        Args:
            event_type: The event type string (use EventTypes constants)
            handler: Callable that receives an Event object
        """
        with self._lock:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        """Remove a handler from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    h for h in self._subscribers[event_type] if h != handler
                ]

    def publish(self, event_type: str, data: dict[str, Any] | None = None):
        """
        Publish an event. All subscribed handlers are called immediately.

        Args:
            event_type: The event type string
            data: Optional dict of event-specific data
        """
        event = Event(event_type=event_type, data=data or {})

        # Store in history
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Get a copy of handlers to call outside the lock
            handlers = list(self._subscribers.get(event_type, []))

        # Call handlers (outside lock to prevent deadlocks)
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"[EventBus] ERROR in handler for {event_type}: {e}")

    def get_recent_events(self, count: int = 20) -> list[Event]:
        """Get the N most recent events."""
        with self._lock:
            return list(self._history[-count:])

    def get_events_by_type(
        self, event_type: str, count: int = 10
    ) -> list[Event]:
        """Get recent events of a specific type."""
        with self._lock:
            matching = [
                e for e in self._history if e.event_type == event_type
            ]
            return matching[-count:]
