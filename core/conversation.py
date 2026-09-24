"""
ULTRON Core — Conversation Controller
=====================================
Orchestrates autonomous conversational flow, proactive greetings, and situational
re-engagements. Ensures ULTRON speaks when visitors arrive, without sounding
robotic or repeating greetings on camera flickers.

Key Responsibilities:
  1. Proactive Greeting: Initiates a calm, witty opening remark when a visitor
     enters view and stays for >= 0.8 seconds.
  2. Cooldown Enforcement: Enforces a 60s cooldown per track ID so visitors
     aren't repeatedly greeted.
  3. Escalating Departure Persuasion: If a person is greeted and does not respond,
     ULTRON re-engages every minute (~60s) with escalating, dry, intimidating remarks
     aimed at convincing them to leave, continuing until they depart.
  4. Non-Blocking Half-Duplex Arbitration: Time-windowed speech detection prevents
     deadlocks from background noise or microphone pops.
"""

import threading
import time
from typing import Optional

import config
from core.event_bus import EventBus, EventTypes, Event
from core.context import ContextManager
from brain.reasoning import ReasoningEngine


class ConversationController:
    """
    Autonomous conversation manager for ULTRON.

    Usage:
        controller = ConversationController(event_bus, context_manager, reasoning_engine)
        controller.start()
        ...
        controller.stop()
    """

    def __init__(
        self,
        event_bus: EventBus,
        context: ContextManager,
        brain: ReasoningEngine,
    ):
        self.event_bus = event_bus
        self.context = context
        self.brain = brain

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Cooldown & timing trackers
        self._last_greeting_time_by_id: dict[int, float] = {}
        self._last_persuasion_time_by_id: dict[int, float] = {}
        self._last_ultron_speech_time: float = 0.0
        self._last_speech_detected_time: float = 0.0
        self._last_user_speech_time: float = 0.0
        self._is_speaking: bool = False

        # Subscriptions
        self.event_bus.subscribe(EventTypes.SPEAKING_STARTED, self._on_speaking_started)
        self.event_bus.subscribe(EventTypes.SPEAKING_FINISHED, self._on_speaking_finished)
        self.event_bus.subscribe(EventTypes.SPEECH_DETECTED, self._on_speech_detected)
        self.event_bus.subscribe(EventTypes.SPEECH_RECOGNIZED, self._on_speech_recognized)
        self.event_bus.subscribe(EventTypes.PERSON_ENTERED, self._on_person_entered)
        self.event_bus.subscribe(EventTypes.PERSON_LEFT, self._on_person_left)

    def start(self):
        """Start the autonomous conversation monitor loop."""
        if self._running:
            return

        self._running = True
        self._last_ultron_speech_time = 0.0
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="UltronConversationController",
            daemon=True,
        )
        self._thread.start()
        print("[ULTRON Conversation] Autonomous conversation flow active (Greetings + Every-Minute Persuasion).")

    def stop(self):
        """Stop the conversation monitor."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        print("[ULTRON Conversation] Conversation controller stopped.")

    # ── Event Callbacks ─────────────────────────────────────────────

    def _on_speaking_started(self, event: Event):
        with self._lock:
            self._is_speaking = True
            self._last_ultron_speech_time = time.time()

    def _on_speaking_finished(self, event: Event):
        with self._lock:
            self._is_speaking = False
            self._last_ultron_speech_time = time.time()

    def _on_speech_detected(self, event: Event):
        # Time-windowed speech detection (self-expires in 3.0s to avoid deadlock on noise)
        with self._lock:
            self._last_speech_detected_time = time.time()

    def _on_speech_recognized(self, event: Event):
        with self._lock:
            now = time.time()
            self._last_speech_detected_time = 0.0
            self._last_user_speech_time = now
            # User spoke! Reset persuasion timers for all currently present visitors
            for tid in list(self._last_persuasion_time_by_id.keys()):
                self._last_persuasion_time_by_id[tid] = now

    def _on_person_entered(self, event: Event):
        track_id = event.data.get("track_id", -1)
        if track_id >= 0:
            with self._lock:
                # Initialize timing for newly entered person
                if track_id not in self._last_persuasion_time_by_id:
                    self._last_persuasion_time_by_id[track_id] = time.time()

    def _on_person_left(self, event: Event):
        track_id = event.data.get("track_id", -1)
        with self._lock:
            # Clean up persuasion timer for person who left
            self._last_persuasion_time_by_id.pop(track_id, None)

    # ── Half-Duplex State Checks ────────────────────────────────────

    def _is_user_actively_talking(self) -> bool:
        """Returns True if VAD detected voice in the last 3.0 seconds."""
        return (time.time() - self._last_speech_detected_time) < 3.0

    def _is_user_recently_spoke(self) -> bool:
        """Returns True if user transcribed speech finished less than 2.0s ago."""
        return (time.time() - self._last_user_speech_time) < 2.0

    # ── Monitor Loop ────────────────────────────────────────────────

    def _monitor_loop(self):
        """Continuous background monitor evaluating conversational triggers."""
        while self._running:
            try:
                time.sleep(0.35)

                if not self._running:
                    break

                now = time.time()

                # Half-duplex arbitration:
                # 1. Don't initiate if ULTRON is currently speaking
                if self._is_speaking:
                    continue

                # 2. Don't initiate if user is actively speaking or just finished
                if self._is_user_actively_talking() or self._is_user_recently_spoke():
                    continue

                # 3. Don't initiate if brain is already processing an LLM call
                if self.brain.is_busy:
                    continue

                persons = self.context.persons
                if not persons:
                    continue

                # ─────────────────────────────────────────────────────────────
                # 1. Autonomous Greeting for New Arrivals
                # ─────────────────────────────────────────────────────────────
                if config.AUTONOMOUS_GREETINGS:
                    for person in persons:
                        if not person.greeting_given:
                            # Confirm person has stayed long enough to establish presence (e.g. 0.8s)
                            if person.dwell_time >= config.GREETING_DELAY_S:
                                last_greeted = self._last_greeting_time_by_id.get(person.track_id, 0.0)
                                time_since_last_ultron = now - self._last_ultron_speech_time

                                # Check cooldown and minimum gap after any prior speech
                                if (now - last_greeted) >= config.GREETING_COOLDOWN_S and time_since_last_ultron >= 2.5:
                                    # Mark greeted in context and update timestamps
                                    self.context.mark_greeting_given(person.track_id)
                                    with self._lock:
                                        self._last_greeting_time_by_id[person.track_id] = now
                                        self._last_persuasion_time_by_id[person.track_id] = now
                                        self._last_ultron_speech_time = now

                                    print(f"[ULTRON Conversation] Triggering greeting for Person ID:{person.track_id} (dwell: {person.dwell_time:.1f}s)")
                                    self.brain.generate_autonomous_greeting(person.track_id)
                                    break  # Only trigger one utterance per loop cycle

                # ─────────────────────────────────────────────────────────────
                # 2. Every-Minute Departure Persuasion (if visitor doesn't respond)
                # ─────────────────────────────────────────────────────────────
                if getattr(config, "PERSUASION_ENABLED", True):
                    persuasion_interval = getattr(config, "PERSUASION_INTERVAL_S", 60.0)

                    for person in persons:
                        # Only persuade persons who have already been greeted
                        if person.greeting_given:
                            last_persuaded = self._last_persuasion_time_by_id.get(person.track_id, person.first_seen)
                            last_interaction = max(last_persuaded, self._last_user_speech_time, self._last_ultron_speech_time)
                            silence_duration = now - last_interaction

                            # If a full minute (~60s) has passed without response
                            if silence_duration >= persuasion_interval:
                                level = self.context.record_persuasion(person.track_id)
                                with self._lock:
                                    self._last_persuasion_time_by_id[person.track_id] = now
                                    self._last_ultron_speech_time = now

                                print(
                                    f"[ULTRON Conversation] Person ID:{person.track_id} silent for {silence_duration:.0f}s. "
                                    f"Delivering departure persuasion (Level {level})..."
                                )
                                self.brain.generate_departure_persuasion(person.track_id, level=level)
                                break  # Only trigger one remark per loop cycle

            except Exception as e:
                print(f"[ULTRON Conversation] Error in monitor loop: {e}")
                time.sleep(1.0)
