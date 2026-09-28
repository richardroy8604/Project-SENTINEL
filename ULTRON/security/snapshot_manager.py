"""
ULTRON Security — Action-Triggered Snapshot Manager
===================================================
Captures, annotates, persists, and broadcasts high-resolution forensic snapshots
whenever significant situational changes or security anomalies occur:

Triggers:
  1. Threat & Weapon Brandishing: Intruder draws or holds a weapon (Crimson Red).
  2. Handheld Object Change: Visitor pulls out or holds a phone (Electric Blue).
  3. Visitor Entry: New person enters the camera's field of view.
  4. Camera Tampering: Blackout / cloth covering lens or defocus spray.
  5. Behavioral Loitering: Person remains stationary past loitering threshold.
  6. Conversational Turn: Visitor speaks or ULTRON formulates a response.
  7. Periodic Telemetry: Periodic visual update during prolonged presence.

Each snapshot is stamped with a tactical HUD banner, saved to disk, and published
as Base64 payloads over WebSockets and IoT MQTT topics.
"""

import base64
import os
import threading
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

import config
from core.event_bus import EventBus, EventTypes, Event
from vision.detector import DetectionResult


# Color hierarchy constants (BGR)
COLOR_WEAPON = (0, 0, 255)       # Crimson Red
COLOR_PHONE = (255, 140, 0)      # Electric Blue
COLOR_PERSON = (0, 255, 100)     # Tactical Green
COLOR_TEXT = (255, 255, 255)     # Crisp White
COLOR_BANNER_BG = (15, 15, 20)   # Deep Charcoal HUD banner


class SnapshotManager:
    """
    Manages forensic incident snapshots, action-triggered frame grabs,
    and image payload broadcasting.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.snapshots_dir = Path(config.SNAPSHOTS_DIR)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._last_snapshot_time: float = 0.0
        self._debounce_s = getattr(config, "SNAPSHOT_DEBOUNCE_S", 1.2)
        self._max_keep = getattr(config, "SNAPSHOTS_MAX_KEEP", 60)

        # Action tracking cache to detect situational changes
        self._last_person_ids: set[int] = set()
        self._last_weapons: dict[int, str] = {}
        self._last_phones: dict[int, bool] = {}
        self._last_security_state: str = config.SecurityState.MONITORING

        # In-memory snapshot cache
        self._latest_snapshot: Optional[dict] = None
        self._recent_snapshots: list[dict] = []
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_detections: Optional[DetectionResult] = None

        # Load existing snapshots on disk to initialize history
        self._load_existing_snapshots()

        # Subscribe to relevant event bus alerts
        self.event_bus.subscribe(EventTypes.WEAPON_DETECTED, self._on_weapon_detected)
        self.event_bus.subscribe(EventTypes.CAMERA_OBSTRUCTED, self._on_camera_obstructed)
        self.event_bus.subscribe(EventTypes.LOITERING_DETECTED, self._on_loitering_detected)
        self.event_bus.subscribe(EventTypes.SPEECH_RECOGNIZED, self._on_speech_recognized)
        self.event_bus.subscribe(EventTypes.RESPONSE_GENERATED, self._on_response_generated)
        self.event_bus.subscribe(EventTypes.STATE_CHANGED, self._on_state_changed)

    def _load_existing_snapshots(self):
        """Pre-populate history with existing images on disk."""
        try:
            files = sorted(
                self.snapshots_dir.glob("*.jpg"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            for f in files[:20]:
                meta = {
                    "filename": f.name,
                    "filepath": str(f),
                    "url": f"/api/snapshots/{f.name}",
                    "timestamp": f.stat().st_mtime,
                    "trigger": f.stem.split("_", 2)[-1].replace("_", " ").upper(),
                    "security_state": "INCIDENT",
                }
                self._recent_snapshots.append(meta)
            if self._recent_snapshots:
                self._latest_snapshot = self._recent_snapshots[0]
        except Exception as e:
            print(f"[ULTRON Snapshot] Warning loading existing snapshots: {e}")

    # ── Real-Time Evaluation from Main Camera Loop ──────────────────

    def update_frame(
        self,
        frame: np.ndarray,
        detections: Optional[DetectionResult],
        security_state: str = config.SecurityState.MONITORING,
    ):
        """
        Evaluates the current video frame against prior situational state.
        Triggers an immediate snapshot if any meaningful action occurs.
        """
        if frame is None:
            return

        with self._lock:
            self._latest_raw_frame = frame
            self._latest_detections = detections

        current_ids: set[int] = set()
        current_weapons: dict[int, str] = {}
        current_phones: dict[int, bool] = {}

        if detections and detections.persons:
            for p in detections.persons:
                current_ids.add(p.track_id)
                if p.holding_weapon:
                    current_weapons[p.track_id] = p.weapon_type
                if p.holding_phone:
                    current_phones[p.track_id] = True

        trigger_reason: Optional[str] = None
        is_critical = False

        # 1. New Weapon Brandished
        for tid, w_type in current_weapons.items():
            if tid not in self._last_weapons or self._last_weapons[tid] != w_type:
                trigger_reason = f"ARMED THREAT: Person ID:{tid} holding {w_type}"
                is_critical = True
                break

        # 2. Phone / Recording Change
        if not trigger_reason:
            for tid in current_phones:
                if not self._last_phones.get(tid, False):
                    trigger_reason = f"OBJECT DETECTED: Person ID:{tid} holding PHONE"
                    break

        # 3. New Visitor Entered
        if not trigger_reason:
            new_ids = current_ids - self._last_person_ids
            if new_ids:
                trigger_reason = f"VISITOR ENTERED: Person ID:{sorted(new_ids)[0]}"

        # 4. Security State Escalation
        if not trigger_reason and security_state != self._last_security_state:
            if security_state in (config.SecurityState.ATTENTION, config.SecurityState.SUSPICIOUS):
                trigger_reason = f"STATE ESCALATION: {self._last_security_state} -> {security_state}"
                is_critical = (security_state == config.SecurityState.SUSPICIOUS)

        # Update cached state
        self._last_person_ids = current_ids
        self._last_weapons = current_weapons
        self._last_phones = current_phones
        self._last_security_state = security_state

        # Dispatch if triggered
        if trigger_reason:
            self.capture_and_dispatch(
                trigger=trigger_reason,
                security_state=security_state,
                critical=is_critical,
                frame_override=frame,
                detections_override=detections,
            )

    # ── Event Callbacks ─────────────────────────────────────────────

    def _on_weapon_detected(self, event: Event):
        tid = event.data.get("track_id", -1)
        w = event.data.get("weapon", "WEAPON")
        self.capture_and_dispatch(
            trigger=f"ARMED THREAT: Person ID:{tid} brandishing {w}",
            security_state=config.SecurityState.SUSPICIOUS,
            critical=True,
        )

    def _on_camera_obstructed(self, event: Event):
        t = event.data.get("type", "LENS_COVERED")
        self.capture_and_dispatch(
            trigger=f"CAMERA TAMPER: {t}",
            security_state=config.SecurityState.SUSPICIOUS,
            critical=True,
        )

    def _on_loitering_detected(self, event: Event):
        tid = event.data.get("track_id", -1)
        dur = event.data.get("duration", 90.0)
        self.capture_and_dispatch(
            trigger=f"LOITERING ALERT: Person ID:{tid} stationary ({dur:.0f}s)",
            security_state=config.SecurityState.SUSPICIOUS,
            critical=False,
        )

    def _on_speech_recognized(self, event: Event):
        text = event.data.get("text", "")
        if text:
            # Capture what intruder was doing when they spoke
            snippet = (text[:30] + "..") if len(text) > 30 else text
            self.capture_and_dispatch(
                trigger=f"INTRUDER SPOKE: \"{snippet}\"",
                critical=False,
            )

    def _on_response_generated(self, event: Event):
        text = event.data.get("text", "")
        if text:
            snippet = (text[:30] + "..") if len(text) > 30 else text
            self.capture_and_dispatch(
                trigger=f"ULTRON REPLIED: \"{snippet}\"",
                critical=False,
            )

    def _on_state_changed(self, event: Event):
        new_s = event.data.get("new_state", "")
        reason = event.data.get("reason", "")
        if new_s in (config.SecurityState.ATTENTION, config.SecurityState.SUSPICIOUS):
            self.capture_and_dispatch(
                trigger=f"SECURITY STATE: {new_s} ({reason})",
                security_state=new_s,
                critical=(new_s == config.SecurityState.SUSPICIOUS),
            )

    # ── Snapshot Capture & Annotation Engine ────────────────────────

    def capture_and_dispatch(
        self,
        trigger: str,
        security_state: str = "",
        critical: bool = False,
        frame_override: Optional[np.ndarray] = None,
        detections_override: Optional[DetectionResult] = None,
    ) -> Optional[dict]:
        """
        Creates an annotated forensic snapshot, encodes it, saves to disk,
        and broadcasts it over EventBus (which forwards to WebSocket and MQTT).
        """
        now = time.time()
        with self._lock:
            # Enforce debounce unless critical (e.g. armed threat or lens tampering)
            if not critical and (now - self._last_snapshot_time) < self._debounce_s:
                return None
            self._last_snapshot_time = now

            frame = frame_override if frame_override is not None else self._latest_raw_frame
            detections = detections_override if detections_override is not None else self._latest_detections

            if frame is None:
                return None

            annotated = frame.copy()

        state = security_state or self._last_security_state
        has_weapon = False

        # ── 1. Render Forensic Overlays ──────────────────────────────
        h, w = annotated.shape[:2]

        # Draw Detection Boxes if detections exist
        if detections:
            # 1. Person Bounding Boxes: Tactical Green
            if detections.persons:
                for p in detections.persons:
                    x1, y1, x2, y2 = p.bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PERSON, 2)

                    label = f"PERSON ID:{p.track_id}"
                    cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + 130, y1), COLOR_PERSON, -1)
                    cv2.putText(
                        annotated, label, (x1 + 4, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
                    )
                    if p.holding_weapon:
                        has_weapon = True

            # 2. Weapon / Threat Boxes: Crimson Red
            if detections.weapon_boxes:
                has_weapon = True
                for item in detections.weapon_boxes:
                    wx1, wy1, wx2, wy2 = item[:4]
                    wname = item[5] if len(item) > 5 else "WEAPON"
                    cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), COLOR_WEAPON, 3)
                    w_tag = f"CRITICAL: {wname}"
                    cv2.rectangle(annotated, (wx1, max(0, wy1 - 24)), (wx1 + 180, wy1), COLOR_WEAPON, -1)
                    cv2.putText(
                        annotated, w_tag, (wx1 + 4, max(16, wy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA
                    )

            # 3. Normal Object / Phone Boxes: Electric Blue
            if detections.phone_boxes:
                for item in detections.phone_boxes:
                    px1, py1, px2, py2 = item[:4]
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_PHONE, 2)
                    cv2.rectangle(annotated, (px1, max(0, py1 - 20)), (px1 + 140, py1), COLOR_PHONE, -1)
                    cv2.putText(
                        annotated, "PHONE RECORDING", (px1 + 4, max(14, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
                    )


        # ── 2. Top Tactical HUD Banner ──────────────────────────────
        banner_h = 36
        banner_overlay = annotated.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), COLOR_BANNER_BG, -1)
        cv2.addWeighted(banner_overlay, 0.85, annotated, 0.15, 0, annotated)

        # Header elements
        state_color = COLOR_WEAPON if state == config.SecurityState.SUSPICIOUS else (
            COLOR_PHONE if state == config.SecurityState.ATTENTION else COLOR_PERSON
        )
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

        cv2.putText(
            annotated, "ULTRON FORENSIC SNAPSHOT", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.putText(
            annotated, f"STATE: {state}", (280, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 2, cv2.LINE_AA
        )
        cv2.putText(
            annotated, time_str, (w - 180, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA
        )

        # Bottom Trigger Subtitle
        sub_overlay = annotated.copy()
        cv2.rectangle(sub_overlay, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(sub_overlay, 0.70, annotated, 0.30, 0, annotated)
        cv2.putText(
            annotated, f"TRIGGER: {trigger}", (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
        )

        # ── 3. Encode & Save ─────────────────────────────────────────
        success, encoded_jpg = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            return None

        jpg_bytes = encoded_jpg.tobytes()
        b64_data = base64.b64encode(jpg_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_data}"

        # Clean filename slug
        slug = "".join(c if c.isalnum() else "_" for c in trigger[:30]).strip("_").lower()
        filename = f"{int(now * 1000)}_{slug}.jpg"
        filepath = self.snapshots_dir / filename

        try:
            with open(filepath, "wb") as f:
                f.write(jpg_bytes)
        except Exception as e:
            print(f"[ULTRON Snapshot] Error saving image to disk: {e}")

        # Meta record
        meta = {
            "filename": filename,
            "filepath": str(filepath),
            "url": f"/api/snapshots/{filename}",
            "timestamp": now,
            "time_str": time_str,
            "trigger": trigger,
            "security_state": state,
            "has_weapon": has_weapon,
            "base64": data_uri,
        }

        with self._lock:
            self._latest_snapshot = meta
            self._recent_snapshots.insert(0, meta)
            if len(self._recent_snapshots) > self._max_keep:
                self._recent_snapshots = self._recent_snapshots[:self._max_keep]

        # Prune old files on disk asynchronously
        threading.Thread(target=self._prune_old_snapshots, daemon=True).start()

        # Publish SNAPSHOT_CAPTURED on EventBus
        self.event_bus.publish(EventTypes.SNAPSHOT_CAPTURED, meta)
        print(f"[ULTRON Snapshot] Action snapshot captured ({filename}) -> \"{trigger}\"")
        return meta

    def _prune_old_snapshots(self):
        """Keep only the latest N snapshot files on disk."""
        try:
            files = sorted(
                self.snapshots_dir.glob("*.jpg"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            for old_file in files[self._max_keep:]:
                try:
                    old_file.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    def get_latest_snapshot(self) -> Optional[dict]:
        """Returns metadata and base64 URI of the latest snapshot."""
        with self._lock:
            return self._latest_snapshot

    def get_recent_snapshots(self, limit: int = 20) -> list[dict]:
        """Returns a list of the most recent snapshots."""
        with self._lock:
            # Strip base64 from list for lightweight JSON responses
            return [
                {k: v for k, v in item.items() if k != "base64"}
                for item in self._recent_snapshots[:limit]
            ]
