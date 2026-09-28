"""
ULTRON Vision — Person Detector & Tracker
==========================================
Wraps YOLO11s for person detection and ByteTrack for multi-person tracking.

Uses ultralytics' built-in model.track() which combines detection + tracking
in a single call. This is cleaner and faster than running them separately.

The detector:
  - Filters for person class only (class 0 in COCO)
  - Returns structured DetectionResult objects
  - Tracks people across frames with persistent IDs
  - Draws detection overlays on frames for the UI
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import cv2

import config

TRACKER_CONFIG = str(Path(__file__).parent / "bytetrack_custom.yaml")


@dataclass
class PersonDetection:
    """A single detected/tracked person in a frame."""

    track_id: int              # Persistent ID across frames (-1 if not tracked)
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    confidence: float          # Detection confidence 0.0 - 1.0
    center: tuple[int, int]    # Center point (cx, cy)
    bbox_area: int             # Bounding box area in pixels
    holding_phone: bool = False  # True if person is holding/recording with a cell phone
    holding_weapon: bool = False # True if person is holding a weapon or dangerous tool
    weapon_type: str = ""        # Name of detected weapon (e.g. "KNIFE", "BLUNT WEAPON")


@dataclass
class DetectionResult:
    """Result of running detection on a single frame."""

    persons: list[PersonDetection] = field(default_factory=list)
    person_count: int = 0
    inference_ms: float = 0.0  # How long detection took
    frame_annotated: np.ndarray | None = None  # Frame with overlays drawn
    weapon_detected: bool = False
    detected_weapons: list[str] = field(default_factory=list)
    phone_boxes: list = field(default_factory=list)
    weapon_boxes: list = field(default_factory=list)


class PersonDetector:
    """
    YOLO11s-based person detector with ByteTrack tracking.

    Usage:
        detector = PersonDetector()
        result = detector.detect(frame)
        print(f"Found {result.person_count} people")
        # result.frame_annotated has bounding boxes drawn
    """

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL,
        confidence: float = config.DETECTION_CONFIDENCE,
        device: str = config.DETECTION_DEVICE,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self._model = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load the YOLO model. Called once at startup.
        Returns True if successful.
        """
        try:
            from ultralytics import YOLO

            print(f"[ULTRON Vision] Loading {self.model_name}...")
            self._model = YOLO(self.model_name)
            self._loaded = True
            print(f"[ULTRON Vision] Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"[ULTRON Vision] ERROR loading model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, frame: np.ndarray, draw: bool = True) -> DetectionResult:
        """
        Detect and track people in a frame.

        Args:
            frame: BGR numpy array from camera
            draw: If True, draw detection overlays on a copy of the frame

        Returns:
            DetectionResult with person list, count, timing, and annotated frame
        """
        if not self._loaded:
            return DetectionResult()

        start_time = time.perf_counter()

        # Classes to detect: 0=person, 67=cell phone, 43=knife, 76=scissors, 34=baseball bat
        classes_to_detect = [0]
        if config.VISION_DETECT_PHONES:
            classes_to_detect.append(67)
        if getattr(config, "VISION_DETECT_WEAPONS", True):
            classes_to_detect.extend(list(config.WEAPON_CLASSES.keys()))

        track_conf = min(self.confidence, getattr(config, "WEAPON_CONFIDENCE", 0.35))

        # Run YOLO tracking (detection + ByteTrack in one call)
        results = self._model.track(
            source=frame,
            classes=classes_to_detect,
            conf=track_conf,
            device=self.device,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - start_time) * 1000

        # Parse results: separate persons, phones, and weapons
        raw_persons = []
        phone_boxes = []
        weapon_boxes = []
        result_obj = results[0] if results else None

        if result_obj and result_obj.boxes is not None and len(result_obj.boxes) > 0:
            boxes = result_obj.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                try:
                    if cls_id == 67:  # Cell phone
                        if conf >= 0.35:
                            phone_boxes.append((x1, y1, x2, y2, conf))
                    elif cls_id in config.WEAPON_CLASSES:
                        if conf >= getattr(config, "WEAPON_CONFIDENCE", 0.35):
                            weapon_name = config.WEAPON_CLASSES[cls_id]
                            weapon_boxes.append((x1, y1, x2, y2, conf, weapon_name))
                    elif cls_id == 0:  # Person
                        if conf >= self.confidence:
                            track_id = -1
                            if boxes.id is not None:
                                try:
                                    track_id = int(boxes.id[i].cpu().numpy())
                                except Exception:
                                    pass
                            if track_id < 0:
                                track_id = 1 + i  # Fallback valid track ID if tracker is initializing

                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            area = (x2 - x1) * (y2 - y1)

                            raw_persons.append({
                                "track_id": track_id,
                                "bbox": (x1, y1, x2, y2),
                                "confidence": conf,
                                "center": (cx, cy),
                                "bbox_area": area,
                            })
                except Exception as box_err:
                    print(f"[ULTRON Vision] Non-fatal error parsing box {i}: {box_err}")

        # Match phones and weapons to persons
        persons = []
        for p in raw_persons:
            px1, py1, px2, py2 = p["bbox"]
            holding_phone = False
            holding_weapon = False
            weapon_type = ""

            # Check phone association (upper 85% of person area)
            for phx1, phy1, phx2, phy2, _ in phone_boxes:
                ph_cx = (phx1 + phx2) // 2
                ph_cy = (phy1 + phy2) // 2
                if (px1 - 25 <= ph_cx <= px2 + 25) and (py1 <= ph_cy <= py1 + int((py2 - py1) * 0.85)):
                    holding_phone = True
                    break

            # Check weapon association (upper / hand / torso region)
            for wx1, wy1, wx2, wy2, _, wname in weapon_boxes:
                wcx = (wx1 + wx2) // 2
                wcy = (wy1 + wy2) // 2
                if (px1 - 35 <= wcx <= px2 + 35) and (py1 <= wcy <= py2 + 30):
                    holding_weapon = True
                    weapon_type = wname
                    break

            persons.append(PersonDetection(
                track_id=p["track_id"],
                bbox=p["bbox"],
                confidence=p["confidence"],
                center=p["center"],
                bbox_area=p["bbox_area"],
                holding_phone=holding_phone,
                holding_weapon=holding_weapon,
                weapon_type=weapon_type,
            ))

        # Draw overlays if requested
        frame_annotated = None
        if draw:
            frame_annotated = self.draw_overlays(frame.copy(), persons, phone_boxes, weapon_boxes)

        detected_weapons = [w[5] for w in weapon_boxes]

        return DetectionResult(
            persons=persons,
            person_count=len(persons),
            inference_ms=inference_ms,
            frame_annotated=frame_annotated,
            weapon_detected=len(weapon_boxes) > 0,
            detected_weapons=detected_weapons,
            phone_boxes=phone_boxes,
            weapon_boxes=weapon_boxes,
        )

    def draw_overlays(
        self,
        frame: np.ndarray,
        persons: list[PersonDetection],
        phone_boxes: list | None = None,
        weapon_boxes: list | None = None,
        loitering_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Draw detection boxes, IDs, phone badges, weapons, and tactical threat brackets."""
        loitering_set = set(loitering_ids or [])

        for person in persons:
            x1, y1, x2, y2 = person.bbox
            track_id = person.track_id
            conf = person.confidence
            holding_phone = person.holding_phone
            holding_weapon = person.holding_weapon
            weapon_type = person.weapon_type
            is_loitering = track_id in loitering_set

            # Color hierarchy: Red for armed threats, Electric Blue for phones, Amber for loitering, Green for normal
            if holding_weapon:
                color = (0, 0, 255)     # Crimson Red (BGR)
                label = f"ID:{track_id} [ARMED: {weapon_type}] {conf:.0%}"
            elif is_loitering:
                color = (0, 165, 255)   # Amber / Orange (BGR)
                label = f"ID:{track_id} [LOITERING] {conf:.0%}"
            elif holding_phone:
                color = (255, 140, 0)   # Electric Blue (BGR)
                label = f"ID:{track_id} [REC PHONE] {conf:.0%}"
            elif track_id >= 0:
                color = (0, 255, 100)   # Tactical Green (BGR)
                label = f"ID:{track_id} {conf:.0%}"
            else:
                color = (0, 255, 255)   # Yellow
                label = f"PERSON {conf:.0%}"

            # Bounding box
            thickness = 3 if holding_weapon else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Tactical corner accents
            corner_len = 16
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Label badge
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 6, y1),
                color,
                -1,
            )

            # Label text (white on red for weapons, black on other colors)
            text_color = (255, 255, 255) if holding_weapon else (0, 0, 0)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                1,
                cv2.LINE_AA,
            )

            # Center dot
            cv2.circle(frame, person.center, 4, color, -1)

        # ── 1. Draw Normal Objects (Phones) in Electric Blue Box ──────
        if phone_boxes:
            blue_color = (255, 140, 0)  # Electric Blue (BGR)
            for phx1, phy1, phx2, phy2, phconf in phone_boxes:
                cv2.rectangle(frame, (phx1, phy1), (phx2, phy2), blue_color, 2)
                ph_label = f"PHONE {phconf:.0%}"
                (pw, ph), _ = cv2.getTextSize(ph_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (phx1, max(0, phy1 - ph - 6)), (phx1 + pw + 4, phy1), blue_color, -1)
                cv2.putText(
                    frame,
                    ph_label,
                    (phx1 + 2, max(ph, phy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # ── 2. Draw Weapons / Sharp / Blunt Tools in Crimson Red Box ──
        if weapon_boxes:
            red_color = (0, 0, 255)  # Crimson Red (BGR)
            for wx1, wy1, wx2, wy2, wconf, wname in weapon_boxes:
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), red_color, 3)
                # Red corner accents for high threat
                w_corner = 12
                cv2.line(frame, (wx1, wy1), (wx1 + w_corner, wy1), red_color, 4)
                cv2.line(frame, (wx1, wy1), (wx1, wy1 + w_corner), red_color, 4)
                cv2.line(frame, (wx2, wy2), (wx2 - w_corner, wy2), red_color, 4)
                cv2.line(frame, (wx2, wy2), (wx2, wy2 - w_corner), red_color, 4)

                w_label = f"THREAT: {wname} {wconf:.0%}"
                (ww, wh), _ = cv2.getTextSize(w_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (wx1, max(0, wy1 - wh - 8)), (wx1 + ww + 6, wy1), red_color, -1)
                cv2.putText(
                    frame,
                    w_label,
                    (wx1 + 3, max(wh, wy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Threat banner if weapon present
        if weapon_boxes:
            banner_text = f"THREAT DETECTED: {weapon_boxes[0][5]}"
            cv2.putText(
                frame,
                banner_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Person count overlay — top-right
        count_text = f"PERSONS: {len(persons)}"
        cv2.putText(
            frame,
            count_text,
            (frame.shape[1] - 180, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    def release(self):
        """Release model resources."""
        self._model = None
        self._loaded = False
        print("[ULTRON Vision] Detector released.")
