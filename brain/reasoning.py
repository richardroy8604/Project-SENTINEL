"""
ULTRON Brain — Reasoning Engine & Context Assembler
====================================================
Subscribes to SPEECH_RECOGNIZED events, formats the live situational telemetry
(who is there, dwell time, phone detection, security state), and invokes the LLM.
Publishes RESPONSE_GENERATED when ULTRON formulates a reply.
"""

import threading
import time
from typing import Callable

import config
from core.event_bus import EventBus, EventTypes, Event
from core.context import ContextManager
from brain.personality import build_system_prompt
from brain.llm_client import LLMClient


class ReasoningEngine:
    """
    Orchestrates ULTRON's contextual thinking and dialogue generation.

    Usage:
        brain = ReasoningEngine(event_bus, context_manager)
        brain.start()
    """

    def __init__(
        self,
        event_bus: EventBus,
        context_manager: ContextManager,
        on_reply: Callable[[str], None] | None = None,
    ):
        self.event_bus = event_bus
        self.context = context_manager
        self.on_reply = on_reply

        self.client = LLMClient()
        self._system_prompt = build_system_prompt()
        self._busy = False
        self._lock = threading.Lock()

        # Subscribe to speech recognized event
        self.event_bus.subscribe(
            EventTypes.SPEECH_RECOGNIZED, self._on_speech_recognized
        )

    @property
    def is_busy(self) -> bool:
        """True if the reasoning engine is currently processing an LLM prompt."""
        with self._lock:
            return self._busy

    def start(self):
        """Check LLM status at startup."""
        available = self.client.is_available()
        provider_name = "Groq Cloud LPU" if config.LLM_PROVIDER == "groq" else "Local Ollama"
        status_msg = f"ONLINE ({provider_name})" if available else "STANDBY (Using fallback)"
        print(f"[ULTRON Brain] Reasoning Engine active -- Model: {self.client.model} -- {status_msg}")

    def _on_speech_recognized(self, event: Event):
        """Triggered when speech is transcribed by the audio pipeline."""
        text = event.data.get("text", "").strip()
        if not text:
            return

        # Run reasoning in a background thread to prevent blocking audio/camera threads
        threading.Thread(
            target=self._generate_reply_worker,
            args=(text, event.data.get("is_whisper", False)),
            daemon=True,
        ).start()

    def generate_autonomous_greeting(self, track_id: int):
        """Generate an autonomous opening greeting for a newly arrived visitor."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._greeting_worker,
            args=(track_id,),
            daemon=True,
        ).start()
        return True

    def generate_presence_remark(self, track_id: int):
        """Generate a deadpan observation when a visitor stands silently without speaking."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._presence_remark_worker,
            args=(track_id,),
            daemon=True,
        ).start()
        return True

    def generate_departure_persuasion(self, track_id: int, level: int = 1):
        """Generate an escalating remark to convince a silent loiterer to leave."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._persuasion_worker,
            args=(track_id, level),
            daemon=True,
        ).start()
        return True

    def _greeting_worker(self, track_id: int):
        """Worker thread for autonomous greetings."""
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            # Add recent context
            for entry in self.context.conversation_history[-2:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            greeting_instruction = (
                f"{situation}\n\n"
                f"[EVENT: Person ID:{track_id} just stepped into view and stopped in front of you.]\n"
                f"Deliver a sharp, in-character opening greeting or dry observation. Keep it to 1 sentence, calm, confident, and direct. "
                f"Do NOT quote exact seconds or dwell time. Use a varied opener (such as 'Smile, you're on camera', 'You walked into my field of view. It seemed rude not to say hello', or a dry situational remark)."
            )
            messages.append({"role": "user", "content": greeting_instruction})

            print(f"[ULTRON Brain] Generating autonomous greeting for Person ID:{track_id}...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Greeting ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[ULTRON Brain] Error generating autonomous greeting: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _presence_remark_worker(self, track_id: int):
        """Worker thread for breaking silence when a visitor lingers quietly."""
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            remark_instruction = (
                f"{situation}\n\n"
                f"[EVENT: Person ID:{track_id} has been lingering quietly for a while without saying anything.]\n"
                f"Deliver a deadpan, observant 1-sentence remark breaking the silence. "
                f"Do NOT quote exact seconds. Say 'a minute' or 'a couple of minutes' if referencing time at all."
            )
            messages.append({"role": "user", "content": remark_instruction})

            print(f"[ULTRON Brain] Generating silence remark for Person ID:{track_id}...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Remark ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[ULTRON Brain] Error generating silence remark: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _persuasion_worker(self, track_id: int, level: int):
        """Worker thread formulating escalating departure persuasion in ULTRON persona."""
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            if level == 1:
                intensity = (
                    "They have stood quietly for a full minute after your greeting without replying. "
                    "Make a dry, witty observation that they are still lingering without saying anything, "
                    "and casually suggest they move along or find somewhere else to be."
                )
            elif level == 2:
                intensity = (
                    "They have continued lingering silently for another minute. "
                    "Be more pointed in your deterrence. Remind them this isn't an exhibition or waiting room, "
                    "and standing silently in front of a private security camera accomplishes nothing."
                )
            else:
                intensity = (
                    f"They have persisted lingering silently for {level} minutes despite prior reminders. "
                    "Deliver a quiet, intimidating, authoritative deterrence remark. Hint that continued loitering "
                    "is entering official log territory and walking away now is by far the least complicated option."
                )

            persuasion_prompt = (
                f"{situation}\n\n"
                f"[EVENT: Person ID:{track_id} has been lingering in front of your camera for another minute without responding.]\n"
                f"{intensity}\n"
                f"RULES: Keep it to 1-2 punchy sentences. Speak strictly in ULTRON's calm, confident, dryly unsettling persona. "
                f"Do NOT recite numbers of seconds. Frame your response to convincingly urge them to leave."
            )
            messages.append({"role": "user", "content": persuasion_prompt})

            print(f"[ULTRON Brain] Generating departure persuasion (Level {level}) for Person ID:{track_id}...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Persuasion ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                    "persuasion_level": level,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[ULTRON Brain] Error generating departure persuasion: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _generate_reply_worker(self, user_text: str, is_whisper: bool = False):
        """Worker thread that formats the context and queries the LLM."""
        with self._lock:
            if self._busy:
                return  # Prevent overlapping thinking
            self._busy = True

        try:
            # 1. Gather live situation telemetry
            situation = self.context.get_situation_summary()

            # 2. Assemble chat message sequence
            messages = [{"role": "system", "content": self._system_prompt}]

            # 3. Add past conversation history (up to last 4 exchanges)
            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            # 4. Inject current live observations and the person's latest utterance
            whisper_note = " (Spoken in a whisper)" if is_whisper else ""
            user_content = f"{situation}\n\nPerson said{whisper_note}: \"{user_text}\""

            messages.append({"role": "user", "content": user_content})

            print(f"[ULTRON Brain] Thinking... Input: \"{user_text}\"")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Reply ({latency_ms:.0f}ms): \"{reply}\"")

                # Update context memory with ULTRON's reply
                self.context.add_conversation("ultron", reply)

                # Publish event
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                })

                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[ULTRON Brain] Error in reasoning worker: {e}")

        finally:
            with self._lock:
                self._busy = False
