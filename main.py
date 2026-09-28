"""
ULTRON — Main Entry Point
===========================
Initializes all modules and runs the main loop.

Stage 1: Camera capture + Dear PyGui dashboard with AI blob.
Stage 2: YOLO11s person detection + ByteTrack tracking.
Stage 3: Event bus + State machine + Context manager.
Stage 4: Audio capture + Silero VAD v5 + faster-whisper STT.
Stage 5: LLM Brain + Persona Engine (Qwen 2.5 7B) + Phone detection.
"""

import sys
import time

import cv2

import config
from setup_models import ensure_vad_model, ensure_tts_models
from core.event_bus import EventBus, EventTypes
from core.state_machine import StateMachine
from core.context import ContextManager
from core.conversation import ConversationController
from core.identity import IdentityTracker
from vision.camera import Camera
from vision.detector import PersonDetector
from audio.listener import AudioListener
from brain.reasoning import ReasoningEngine
from voice import TTSEngine, VoicePlayback
from ui.dashboard import Dashboard
from security import CameraTamperDetector, LoiteringDetector


def main():
    print("=" * 60)
    print("  U L T R O N  —  AI Security System")
    print("=" * 60)
    print()

    # ── Ensure Required Models ──────────────────────────────────────
    print("[ULTRON] Verifying model dependencies...")
    ensure_vad_model()
    ensure_tts_models()

    # ── Initialize Core Systems ─────────────────────────────────────
    print("[ULTRON] Initializing core systems...")
    event_bus = EventBus()
    state_machine = StateMachine(event_bus)
    context = ContextManager(event_bus)

    # ── Initialize Behavior & Security Analytics (Stage 8) ─────────
    print("[ULTRON] Initializing security & behavior analytics...")
    tamper_detector = CameraTamperDetector(event_bus=event_bus)
    loitering_detector = LoiteringDetector(event_bus=event_bus)

    # ── Initialize Camera ───────────────────────────────────────────
    print("[ULTRON] Initializing camera...")
    camera = Camera()
    if not camera.start():
        print("[ULTRON] FATAL: Cannot open camera. Exiting.")
        sys.exit(1)

    time.sleep(0.5)

    # ── Initialize Person & Phone Detector ──────────────────────────
    print("[ULTRON] Initializing vision detector...")
    detector = PersonDetector()
    if not detector.load():
        print("[ULTRON] WARNING: Detector failed to load.")
        detector = None

    # ── Initialize Dashboard ────────────────────────────────────────
    print("[ULTRON] Initializing dashboard...")
    dashboard = Dashboard()
    dashboard.setup()

    # ── Initialize Voice Output (Kokoro TTS) ────────────────────────
    print("[ULTRON] Initializing voice output (Kokoro TTS)...")
    voice_playback = VoicePlayback(
        event_bus=event_bus,
        on_level=dashboard.update_audio_level,
    )
    if not voice_playback.start():
        print("[ULTRON] WARNING: Voice playback failed to start.")
        voice_playback = None

    # ── Initialize Audio Pipeline (VAD + STT) ───────────────────────
    print("[ULTRON] Initializing audio pipeline...")
    audio_listener = AudioListener(
        event_bus=event_bus,
        on_level=dashboard.update_audio_level,
    )
    if not audio_listener.start():
        print("[ULTRON] WARNING: Audio pipeline failed to start.")
        audio_listener = None

    # ── Initialize LLM Brain (Reasoning Engine) ─────────────────────
    print("[ULTRON] Initializing LLM brain & personality...")
    brain = ReasoningEngine(
        event_bus=event_bus,
        context_manager=context,
        on_reply=dashboard.update_ultron_reply,
    )
    brain.start()

    # ── Initialize Autonomous Conversation Controller ──────────────
    print("[ULTRON] Initializing autonomous conversation flow...")
    conversation = ConversationController(
        event_bus=event_bus,
        context=context,
        brain=brain,
    )
    conversation.start()

    # Wire dashboard to event bus for logging and state visualization
    def on_state_changed(event):
        old = event.data.get("old_state", "?")
        new = event.data.get("new_state", "?")
        reason = event.data.get("reason", "")
        dashboard.log_event(f"[STATE] {old} -> {new} ({reason})")
        dashboard.set_security_state(new)

    def on_speech_detected(event):
        prob = event.data.get("probability", 0.0)
        dashboard.log_event(f"[AUDIO] Speech detected (prob: {prob:.0%})")

    def on_speech_recognized(event):
        text = event.data.get("text", "")
        latency = event.data.get("latency_ms", 0.0)
        dashboard.update_last_speech(text)
        dashboard.log_event(f"[SPEECH] Heard: \"{text}\" ({latency:.0f}ms)")

    def on_speaking_started(event):
        dashboard.set_blob_speaking(True, intensity=1.0)
        dashboard.log_event("[VOICE] Speaking...")

    def on_speaking_finished(event):
        dashboard.set_blob_speaking(False)
        dashboard.update_audio_level(0.0)

    def on_response_generated(event):
        reply = event.data.get("text", "")
        latency = event.data.get("latency_ms", 0.0)
        dashboard.update_ultron_reply(reply)
        dashboard.log_event(f"[ULTRON] \"{reply}\" ({latency:.0f}ms)")
        if voice_playback:
            voice_playback.speak(reply)

    def on_weapon_detected(event):
        tid = event.data.get("track_id", -1)
        w = event.data.get("weapon", "WEAPON")
        dashboard.log_event(f"[ALERT] ARMED THREAT: Person ID:{tid} holding {w}!")

    def on_loitering_detected(event):
        tid = event.data.get("track_id", -1)
        dur = event.data.get("duration", 0)
        dashboard.log_event(f"[ALERT] Loitering: Person ID:{tid} stationary ({int(dur)}s)")

    def on_camera_obstructed(event):
        t = event.data.get("type", "UNKNOWN")
        dashboard.log_event(f"[ALERT] Camera tamper detected: {t}")

    def on_tamper_cleared(event):
        t = event.data.get("cleared_type", "")
        dashboard.log_event(f"[SECURITY] Camera tamper cleared ({t})")

    def on_whisper_detected(event):
        conf = event.data.get("confidence", 0.0)
        dashboard.log_event(f"[AUDIO] Whispered speech detected (conf: {conf:.0%})")

    event_bus.subscribe(EventTypes.STATE_CHANGED, on_state_changed)
    event_bus.subscribe(EventTypes.SPEECH_DETECTED, on_speech_detected)
    event_bus.subscribe(EventTypes.SPEECH_RECOGNIZED, on_speech_recognized)
    event_bus.subscribe(EventTypes.RESPONSE_GENERATED, on_response_generated)
    event_bus.subscribe(EventTypes.SPEAKING_STARTED, on_speaking_started)
    event_bus.subscribe(EventTypes.SPEAKING_FINISHED, on_speaking_finished)
    event_bus.subscribe(EventTypes.WEAPON_DETECTED, on_weapon_detected)
    event_bus.subscribe(EventTypes.LOITERING_DETECTED, on_loitering_detected)
    event_bus.subscribe(EventTypes.CAMERA_OBSTRUCTED, on_camera_obstructed)
    event_bus.subscribe(EventTypes.TAMPER_CLEARED, on_tamper_cleared)
    event_bus.subscribe(EventTypes.WHISPER_DETECTED, on_whisper_detected)

    dashboard.log_event("[ULTRON] Core systems online.")
    dashboard.log_event("[ULTRON] Camera active.")
    if detector and detector.is_loaded:
        phone_tag = " + Phone Detection" if config.VISION_DETECT_PHONES else ""
        weapon_tag = " + Threat/Weapon Recognition" if getattr(config, "VISION_DETECT_WEAPONS", True) else ""
        dashboard.log_event(f"[ULTRON] Vision: YOLO11s on {config.DETECTION_DEVICE}{phone_tag}{weapon_tag}")
    if audio_listener:
        dashboard.log_event(f"[ULTRON] Audio: Silero VAD + faster-whisper ({config.STT_MODEL_SIZE}) + Whisper Classifier")
    dashboard.log_event(f"[ULTRON] Brain: Persona active ({config.LLM_MODEL})")
    if getattr(config, "LOITERING_DETECTION_ENABLED", True):
        dashboard.log_event(f"[ULTRON] Behavior: Loitering engine active ({int(config.LOITERING_THRESHOLD_S)}s limit)")
    if getattr(config, "TAMPER_DETECTION_ENABLED", True):
        dashboard.log_event("[ULTRON] Security: Camera tamper & obstruction monitor active")
    if voice_playback:
        if config.TTS_PROVIDER == "fish":
            v_name = f"Fish Audio ({config.FISH_AUDIO_VOICE_ID[:8]}...)"
        elif config.TTS_PROVIDER == "edge":
            v_name = config.TTS_EDGE_VOICE.replace("en-US-", "").replace("Neural", "")
        else:
            v_name = config.TTS_VOICE
        dashboard.log_event(f"[ULTRON] Voice: {v_name} ({config.TTS_PROVIDER.upper()})")
    dashboard.log_event("[ULTRON] Flow: Autonomous greeting & re-engagement active")
    dashboard.log_event("[ULTRON] Monitoring...")

    # ── Main Loop ───────────────────────────────────────────────────
    print("[ULTRON] System online. Close the window to exit.")
    print()

    # Identity tracker maintains stable canonical IDs across departures and returns
    identity_tracker = IdentityTracker(reid_window_s=60.0)
    active_tracked_ids: set[int] = set()
    tracked_last_seen: dict[int, float] = {}

    try:
        while True:
            # 1. Get latest camera frame
            frame_bgr = camera.get_frame()
            if frame_bgr is None:
                if not dashboard.render_frame():
                    break
                continue

            # 2. Evaluate physical camera tampering / obstruction
            is_tampered, tamper_type, _ = tamper_detector.evaluate_frame(frame_bgr)

            # 3. Run person, weapon, & phone detection + tracking
            if detector and detector.is_loaded:
                # Run detection without drawing yet to resolve canonical IDs first
                result = detector.detect(frame_bgr, draw=False)

                now_ts = time.time()

                # ── Map raw tracker IDs to persistent canonical IDs ──
                for p in result.persons:
                    if p.track_id >= 0:
                        p.track_id = identity_tracker.resolve_track_id(p.track_id, now_ts)

                current_ids = {
                    p.track_id for p in result.persons if p.track_id >= 0
                }
                identity_tracker.set_active_ids(current_ids)

                # Update last seen for all detected IDs
                for tid in current_ids:
                    tracked_last_seen[tid] = now_ts

                # New persons entering
                for tid in current_ids - active_tracked_ids:
                    active_tracked_ids.add(tid)
                    event_bus.publish(EventTypes.PERSON_ENTERED, {
                        "track_id": tid,
                    })
                    dashboard.log_event(
                        f"[VISION] Person ID:{tid} entered view."
                    )

                # Check for persons who truly left (not seen for >= PERSON_LOST_GRACE_S)
                for tid in list(active_tracked_ids):
                    if tid not in current_ids:
                        last_seen_time = tracked_last_seen.get(tid, 0.0)
                        if (now_ts - last_seen_time) >= config.PERSON_LOST_GRACE_S:
                            active_tracked_ids.remove(tid)
                            tracked_last_seen.pop(tid, None)
                            identity_tracker.on_person_departed(tid, now_ts)

                            # Calculate how long they were present
                            person_ctx = None
                            for p in context.persons:
                                if p.track_id == tid:
                                    person_ctx = p
                                    break
                            duration = (
                                int(person_ctx.dwell_time) if person_ctx else 0
                            )
                            event_bus.publish(EventTypes.PERSON_LEFT, {
                                "track_id": tid,
                                "duration": duration,
                            })
                            dashboard.log_event(
                                f"[VISION] Person ID:{tid} left view ({duration}s)."
                            )

                # ── Publish Weapon Detected Events ──
                if result.weapon_detected:
                    for p in result.persons:
                        if p.holding_weapon and p.track_id >= 0:
                            event_bus.publish(EventTypes.WEAPON_DETECTED, {
                                "track_id": p.track_id,
                                "weapon": p.weapon_type,
                                "timestamp": now_ts,
                            })

                # ── Update Loitering Analytics ──
                loitering_ids = loitering_detector.update_tracks(
                    result.persons,
                    frame_width=frame_bgr.shape[1],
                    frame_height=frame_bgr.shape[0],
                )

                # Update context with current persons & holding phone / weapon state
                for person in result.persons:
                    if person.track_id >= 0:
                        context.update_person(
                            person.track_id,
                            holding_phone=person.holding_phone,
                            holding_weapon=person.holding_weapon,
                            weapon_type=person.weapon_type,
                        )

                # Sync person count to state machine using active_tracked_ids
                state_machine.set_person_count(len(active_tracked_ids))

                # Update dashboard detection stats
                dashboard.update_detection_info(
                    person_count=result.person_count,
                    inference_ms=result.inference_ms,
                )

                # Draw overlays with persistent canonical IDs and loitering badges
                display_frame = detector.draw_overlays(
                    frame_bgr.copy(),
                    result.persons,
                    phone_boxes=result.phone_boxes,
                    weapon_boxes=result.weapon_boxes,
                    loitering_ids=loitering_ids,
                )

                # Overlay tamper alert if actively obstructed
                if is_tampered:
                    cv2.putText(
                        display_frame,
                        f"CAMERA OBSTRUCTED: {tamper_type}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            else:
                display_frame = frame_bgr.copy()
                if is_tampered:
                    cv2.putText(
                        display_frame,
                        f"CAMERA OBSTRUCTED: {tamper_type}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # 3. Update video display
            dashboard.update_video(display_rgb)
            dashboard.update_camera_fps(camera.fps)

            # 4. Render UI frame (includes blob animation)
            if not dashboard.render_frame():
                break

    except KeyboardInterrupt:
        print("\n[ULTRON] Keyboard interrupt received.")

    finally:
        print("[ULTRON] Shutting down...")
        if conversation:
            conversation.stop()
        if voice_playback:
            voice_playback.stop()
        if audio_listener:
            audio_listener.stop()
        camera.stop()
        if detector:
            detector.release()
        dashboard.shutdown()
        print("[ULTRON] System offline.")


if __name__ == "__main__":
    main()
