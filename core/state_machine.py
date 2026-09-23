"""
ULTRON Core — Security State Machine
======================================
Deterministic state machine that manages ULTRON's security posture.

States:
    IDLE        → System disarmed
    MONITORING  → Armed, no persons detected
    ATTENTION   → Person(s) detected, actively tracking
    SUSPICIOUS  → Anomaly detected (loitering, tampering, whisper, etc.)

Transitions are rule-based and deterministic — the LLM never controls
the security state. It only reads it as context.
"""

import time
import threading

import config
from core.event_bus import EventBus, EventTypes, Event


class StateMachine:
    """
    Deterministic security state machine.

    Transitions are triggered by events from the event bus.
    When the state changes, a STATE_CHANGED event is published.
    """

    def __init__(self, event_bus: EventBus):
        self._bus = event_bus
        self._state = config.SecurityState.MONITORING
        self._previous_state = config.SecurityState.IDLE
        self._state_since = time.time()
        self._lock = threading.Lock()

        # Anomaly tracking
        self._active_anomalies: dict[str, str] = {}  # {anomaly_id: reason}
        self._person_count = 0

        # Subscribe to events that can trigger state changes
        self._bus.subscribe(EventTypes.PERSON_ENTERED, self._on_person_entered)
        self._bus.subscribe(EventTypes.PERSON_LEFT, self._on_person_left)
        self._bus.subscribe(EventTypes.LOITERING_DETECTED, self._on_loitering)
        self._bus.subscribe(EventTypes.CAMERA_OBSTRUCTED, self._on_camera_obstructed)
        self._bus.subscribe(EventTypes.WHISPER_DETECTED, self._on_whisper)

    @property
    def state(self) -> str:
        """Current security state."""
        with self._lock:
            return self._state

    @property
    def state_duration(self) -> float:
        """Seconds since last state change."""
        return time.time() - self._state_since

    @property
    def anomalies(self) -> dict[str, str]:
        """Active anomalies causing SUSPICIOUS state."""
        with self._lock:
            return dict(self._active_anomalies)

    def _change_state(self, new_state: str, reason: str):
        """
        Transition to a new state if different from current.
        Publishes STATE_CHANGED event.
        """
        with self._lock:
            if new_state == self._state:
                return

            old_state = self._state
            self._previous_state = old_state
            self._state = new_state
            self._state_since = time.time()

        print(f"[ULTRON State] {old_state} -> {new_state} ({reason})")

        self._bus.publish(EventTypes.STATE_CHANGED, {
            "old_state": old_state,
            "new_state": new_state,
            "reason": reason,
        })

    def _evaluate_state(self):
        """
        Re-evaluate what state we should be in based on current conditions.
        Called after any event that might change state.
        """
        with self._lock:
            has_anomalies = len(self._active_anomalies) > 0
            has_persons = self._person_count > 0

        if has_anomalies:
            reasons = ", ".join(self._active_anomalies.values())
            self._change_state(config.SecurityState.SUSPICIOUS, reasons)
        elif has_persons:
            self._change_state(
                config.SecurityState.ATTENTION, "Person(s) present"
            )
        else:
            self._change_state(
                config.SecurityState.MONITORING, "Area clear"
            )

    # ── Event Handlers ──────────────────────────────────────────────

    def _on_person_entered(self, event: Event):
        """Person entered the view."""
        with self._lock:
            self._person_count = max(1, self._person_count + 1)
        self._evaluate_state()

    def _on_person_left(self, event: Event):
        """Person left the view."""
        track_id = event.data.get("track_id", -1)
        with self._lock:
            self._person_count = max(0, self._person_count - 1)
            # Clear any anomalies tied to this person
            anomaly_key = f"loiter_{track_id}"
            self._active_anomalies.pop(anomaly_key, None)
        self._evaluate_state()

    def _on_loitering(self, event: Event):
        """Loitering detected for a person."""
        track_id = event.data.get("track_id", -1)
        duration = event.data.get("duration", 0)
        with self._lock:
            self._active_anomalies[f"loiter_{track_id}"] = (
                f"Person ID:{track_id} loitering ({int(duration)}s)"
            )
        self._evaluate_state()

    def _on_camera_obstructed(self, event: Event):
        """Camera tamper/obstruction detected."""
        obstruction_type = event.data.get("type", "unknown")
        with self._lock:
            self._active_anomalies["camera_tamper"] = (
                f"Camera obstruction ({obstruction_type})"
            )
        self._evaluate_state()

    def _on_whisper(self, event: Event):
        """Whispered speech detected."""
        with self._lock:
            self._active_anomalies["whisper"] = "Whispered speech detected"
        self._evaluate_state()

    def clear_anomaly(self, anomaly_id: str):
        """Manually clear a specific anomaly."""
        with self._lock:
            self._active_anomalies.pop(anomaly_id, None)
        self._evaluate_state()

    def set_person_count(self, count: int):
        """Directly set person count (called from main loop for sync)."""
        with self._lock:
            self._person_count = count
