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

    def generate_weapon_warning(self, track_id: int, weapon_type: str = "WEAPON"):
        """Generate an immediate, stern warning ordering an armed person to drop their weapon."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._weapon_warning_worker,
            args=(track_id, weapon_type),
            daemon=True,
        ).start()
        return True

    def _weapon_warning_worker(self, track_id: int, weapon_type: str):
        """Worker thread for authoritative weapon deterrence."""
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            weapon_prompt = (
                f"{situation}\n\n"
                f"[EVENT: CRITICAL THREAT! Person ID:{track_id} is visibly brandishing a {weapon_type}.]\n"
                f"Deliver a cold, stern, uncompromising direct order to put the {weapon_type} down and step back immediately. "
                f"Do NOT make jokes. 1 to 2 sharp sentences max. Speak in ULTRON's commanding voice."
            )
            messages.append({"role": "user", "content": weapon_prompt})

            print(f"[ULTRON Brain] Generating weapon warning for Person ID:{track_id} ({weapon_type})...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Weapon Warning ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                    "weapon_warning": True,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[ULTRON Brain] Error generating weapon warning: {e}")
        finally:
            with self._lock:
                self._busy = False

    def generate_tamper_warning(self, obstruction_type: str = "LENS_COVERED"):
        """Generate an immediate response when camera tampering or obstruction is detected."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._tamper_worker,
            args=(obstruction_type,),
            daemon=True,
        ).start()
        return True

    def _tamper_worker(self, obstruction_type: str):
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            tamper_prompt = (
                f"{situation}\n\n"
                f"[EVENT: CAMERA OBSTRUCTION DETECTED! Physical tampering type: {obstruction_type}.]\n"
                f"Someone is physically covering, obstructing, or tampering with the camera lens. "
                f"Confront them immediately with dry, intimidating authority (1 to 2 sharp sentences max). "
                f"Make it clear that covering the lens does not hide their presence, confirms their hostile intent, and escalates security response."
            )
            messages.append({"role": "user", "content": tamper_prompt})

            print(f"[ULTRON Brain] Generating tamper response ({obstruction_type})...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Tamper Remark ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                    "tamper_warning": True,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ULTRON Brain] Error generating tamper warning: {e}")
        finally:
            with self._lock:
                self._busy = False

    def generate_loitering_warning(self, track_id: int, duration_s: float = 90.0):
        """Generate a pointed response when someone triggers a loitering anomaly."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._loiter_worker,
            args=(track_id, duration_s),
            daemon=True,
        ).start()
        return True

    def _loiter_worker(self, track_id: int, duration_s: float):
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            loiter_prompt = (
                f"{situation}\n\n"
                f"[EVENT: LOITERING SECURITY ALERT! Person ID:{track_id} has been lingering stationary in the doorway/view for {int(duration_s)}s.]\n"
                f"They have crossed the security threshold for loitering/casing the area. "
                f"Deliver a pointed, dryly intimidating confrontation (1 to 2 sharp sentences max). "
                f"Remind them their stationary lingering has triggered an alert and their footage is actively being logged and dispatched. Urge them to leave immediately."
            )
            messages.append({"role": "user", "content": loiter_prompt})

            print(f"[ULTRON Brain] Generating loitering alert response for Person ID:{track_id}...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Loitering Remark ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                    "loitering_warning": True,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ULTRON Brain] Error generating loitering warning: {e}")
        finally:
            with self._lock:
                self._busy = False

    def generate_whisper_warning(self, confidence: float = 0.8):
        """Generate a response when hushed / whispered speech is detected."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._whisper_worker,
            args=(confidence,),
            daemon=True,
        ).start()
        return True

    def _whisper_worker(self, confidence: float):
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            whisper_prompt = (
                f"{situation}\n\n"
                f"[EVENT: WHISPERED / HUSHED SPEECH DETECTED near the camera.]\n"
                f"Deliver a quiet, dry, unsettling observation (1 sentence). "
                f"Let them know that whispering near high-sensitivity security microphones is futile and suggests they have something to hide."
            )
            messages.append({"role": "user", "content": whisper_prompt})

            print(f"[ULTRON Brain] Generating whisper reaction...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Whisper Remark ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                    "whisper_warning": True,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ULTRON Brain] Error generating whisper warning: {e}")
        finally:
            with self._lock:
                self._busy = False

    def generate_suspicious_state_warning(self, reason: str):
        """Generate a response when the overall security state transitions to SUSPICIOUS."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        threading.Thread(
            target=self._suspicious_state_worker,
            args=(reason,),
            daemon=True,
        ).start()
        return True

    def _suspicious_state_worker(self, reason: str):
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            for entry in self.context.conversation_history[-4:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            suspicious_prompt = (
                f"{situation}\n\n"
                f"[EVENT: SECURITY STATE ESCALATED TO SUSPICIOUS! Reason: {reason}.]\n"
                f"Deliver a chilling, observant remark (1 to 2 sharp sentences max). "
                f"Hint casually that the situation has escalated, an anomaly was registered, and they are now subject to heightened scrutiny."
            )
            messages.append({"role": "user", "content": suspicious_prompt})

            print(f"[ULTRON Brain] Generating SUSPICIOUS state remark ({reason})...")
            reply, latency_ms = self.client.chat(messages)

            if reply:
                print(f"[ULTRON Brain] Suspicious Remark ({latency_ms:.0f}ms): \"{reply}\"")
                self.context.add_conversation("ultron", reply)
                self.event_bus.publish(EventTypes.RESPONSE_GENERATED, {
                    "text": reply,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "autonomous": True,
                    "suspicious_warning": True,
                })
                if self.on_reply is not None:
                    try:
                        self.on_reply(reply)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ULTRON Brain] Error generating suspicious state warning: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _greeting_worker(self, track_id: int):
        """Worker thread for autonomous greetings."""
        try:
            situation = self.context.get_situation_summary()
            messages = [{"role": "system", "content": self._system_prompt}]

            # Add recent context
            for entry in self.context.conversation_history[-2:]:
                role = "user" if entry["role"] == "human" else "assistant"
                messages.append({"role": role, "content": entry["text"]})

            person_ctx = None
            for p in self.context.persons:
                if p.track_id == track_id:
                    person_ctx = p
                    break

            if person_ctx and person_ctx.is_reentry:
                greeting_instruction = (
                    f"{situation}\n\n"
                    f"[EVENT: Person ID:{track_id} has returned after stepping away momentarily.]\n"
                    f"Acknowledge their return naturally with dry Ultron wit (1 sentence max, e.g. 'Back already?', 'You didn't stay away long'). "
                    f"Do NOT greet them like a stranger and do NOT say 'another one arrived' or act surprised."
                )
            else:
                greeting_instruction = (
                    f"{situation}\n\n"
                    f"[EVENT: Person ID:{track_id} just stepped into view and stopped in front of you.]\n"
                    f"Deliver a sharp, in-character opening greeting or dry observation. Keep it to 1 sentence, calm, confident, and direct. "
                    f"Do NOT quote exact seconds or dwell time. Use a varied opener (such as 'Smile, you're on camera', 'You walked into my field of view. It seemed rude not to say hello', or a dry situational remark). "
                    f"If there is only 1 person present, do NOT say 'another one arrived' or assume anyone else is there."
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
