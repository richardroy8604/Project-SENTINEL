"""
ULTRON Core — Conversation Controller
=====================================
Orchestrates autonomous conversational flow, proactive greetings, and situational
re-engagements. Ensures ULTRON speaks when visitors arrive, without sounding
robotic or repeating greetings on camera flickers.

Key Responsibilities:
  1. Proactive Greeting: Initiates a calm, witty opening remark when a visitor
     enters view and stays for >= 1.5 seconds.
  2. Cooldown Enforcement: Enforces a 60s cooldown per track ID so visitors
     aren't repeatedly greeted.
  3. Silence Re-engagement: If a visitor stands silently for > 22 seconds without
     speaking, delivers a deadpan observation breaking the silence (at most once).
  4. Half-Duplex Arbitration: Suppresses autonomous initiations while ULTRON is
     actively speaking or while the user is talking.
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

        # Cooldown & interaction tracking
        self._last_greeting_time_by_id: dict[int, float] = {}
        self._last_global_interaction_time: float = time.time()
        self._is_speaking: bool = False
        self._is_user_talking: bool = False
        self._last_user_speech_end: float = 0.0

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
        self._last_global_interaction_time = time.time()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="UltronConversationController",
            daemon=True,
        )
        self._thread.start()
        print("[ULTRON Conversation] Autonomous conversation flow active.")

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
            self._last_global_interaction_time = time.time()

    def _on_speaking_finished(self, event: Event):
        with self._lock:
            self._is_speaking = False
            self._last_global_interaction_time = time.time()

    def _on_speech_detected(self, event: Event):
        with self._lock:
            self._is_user_talking = True

    def _on_speech_recognized(self, event: Event):
        with self._lock:
            self._is_user_talking = False
            self._last_user_speech_end = time.time()
            self._last_global_interaction_time = time.time()

    def _on_person_entered(self, event: Event):
        track_id = event.data.get("track_id", -1)
        # Note: We do NOT immediately fire greeting on enter; we wait GREETING_DELAY_S (1.5s)
        # in the monitor loop to confirm stable presence.

    def _on_person_left(self, event: Event):
        track_id = event.data.get("track_id", -1)
        # Retain greeting timestamp in cooldown dict so immediate re-entry doesn't re-greet

    # ── Monitor Loop ────────────────────────────────────────────────

    def _monitor_loop(self):
        """Continuous background monitor evaluating conversational triggers."""
        while self._running:
            try:
                time.sleep(0.4)

                if not self._running:
                    break

                # Half-duplex check: do not initiate if ULTRON is speaking or user is talking
                with self._lock:
                    if self._is_speaking or self._is_user_talking:
                        continue
                    if (time.time() - self._last_user_speech_end) < 2.0:
                        continue  # Wait 2 seconds after user finishes speaking before autonomous action

                if self.brain.is_busy:
                    continue

                now = time.time()
                persons = self.context.persons

                if not persons:
                    continue

                # 1. Check for autonomous greetings (new arrivals)
                if config.AUTONOMOUS_GREETINGS:
                    for person in persons:
                        if not person.greeting_given:
                            # Confirm person has stayed long enough to establish stable presence
                            if person.dwell_time >= config.GREETING_DELAY_S:
                                last_greeted = self._last_greeting_time_by_id.get(person.track_id, 0.0)
                                # Enforce per-person cooldown & minimum gap between any greetings
                                if (now - last_greeted) >= config.GREETING_COOLDOWN_S and (now - self._last_global_interaction_time) >= 6.0:
                                    # Mark greeted in context and record timestamp
                                    self.context.mark_greeting_given(person.track_id)
                                    with self._lock:
                                        self._last_greeting_time_by_id[person.track_id] = now
                                        self._last_global_interaction_time = now

                                    print(f"[ULTRON Conversation] Triggering greeting for Person ID:{person.track_id} (dwell: {person.dwell_time:.1f}s)")
                                    self.brain.generate_autonomous_greeting(person.track_id)
                                    break  # Only trigger one greeting per loop iteration

                # 2. Check for silence re-engagement (lingering without speaking)
                if config.SILENCE_REENGAGE_ENABLED:
                    for person in persons:
                        if person.greeting_given and not person.remark_given:
                            silence_duration = now - self._last_global_interaction_time
                            if person.dwell_time >= config.SILENCE_REENGAGE_TIMEOUT_S and silence_duration >= config.SILENCE_REENGAGE_TIMEOUT_S:
                                self.context.mark_remark_given(person.track_id)
                                with self._lock:
                                    self._last_global_interaction_time = now

                                print(f"[ULTRON Conversation] Triggering silence re-engagement for Person ID:{person.track_id} (silence: {silence_duration:.1f}s)")
                                self.brain.generate_presence_remark(person.track_id)
                                break  # Only trigger one remark per loop iteration

            except Exception as e:
                print(f"[ULTRON Conversation] Error in monitor loop: {e}")
                time.sleep(1.0)
