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
import numpy as np
import cv2

import config


@dataclass
class PersonDetection:
    """A single detected/tracked person in a frame."""

    track_id: int              # Persistent ID across frames (-1 if not tracked)
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    confidence: float          # Detection confidence 0.0 - 1.0
    center: tuple[int, int]    # Center point (cx, cy)
    bbox_area: int             # Bounding box area in pixels


@dataclass
class DetectionResult:
    """Result of running detection on a single frame."""

    persons: list[PersonDetection] = field(default_factory=list)
    person_count: int = 0
    inference_ms: float = 0.0  # How long detection took
    frame_annotated: np.ndarray | None = None  # Frame with overlays drawn


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

        # Run YOLO tracking (detection + ByteTrack in one call)
        # classes=[0] filters for "person" only
        # persist=True keeps track IDs across frames
        results = self._model.track(
            source=frame,
            classes=[0],                      # Person class only
            conf=self.confidence,
            device=self.device,
            persist=True,                     # Persistent tracking
            tracker="bytetrack.yaml",         # ByteTrack algorithm
            verbose=False,                    # No console spam
        )

        inference_ms = (time.perf_counter() - start_time) * 1000

        # Parse results
        persons = []
        result_obj = results[0] if results else None

        if result_obj and result_obj.boxes is not None and len(result_obj.boxes) > 0:
            boxes = result_obj.boxes

            for i in range(len(boxes)):
                # Bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                # Track ID (-1 if tracking failed for this detection)
                track_id = -1
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                # Confidence
                conf = float(boxes.conf[i].cpu().numpy())

                # Center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Area
                area = (x2 - x1) * (y2 - y1)

                persons.append(PersonDetection(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(cx, cy),
                    bbox_area=area,
                ))

        # Draw overlays if requested
        frame_annotated = None
        if draw:
            frame_annotated = self._draw_overlays(frame.copy(), persons)

        return DetectionResult(
            persons=persons,
            person_count=len(persons),
            inference_ms=inference_ms,
            frame_annotated=frame_annotated,
        )

    def _draw_overlays(
        self, frame: np.ndarray, persons: list[PersonDetection]
    ) -> np.ndarray:
        """Draw detection boxes, IDs, and confidence on the frame."""

        for person in persons:
            x1, y1, x2, y2 = person.bbox
            track_id = person.track_id
            conf = person.confidence

            # Color: green for tracked, yellow for untracked
            if track_id >= 0:
                color = (0, 255, 100)  # Green (BGR)
                label = f"ID:{track_id} {conf:.0%}"
            else:
                color = (0, 255, 255)  # Yellow
                label = f"PERSON {conf:.0%}"

            # Bounding box — slightly thick for visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents (tactical look)
            corner_len = 15
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Label background
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

            # Label text (black on colored background)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Center dot
            cv2.circle(frame, person.center, 4, color, -1)

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
