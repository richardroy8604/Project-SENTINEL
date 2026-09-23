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
