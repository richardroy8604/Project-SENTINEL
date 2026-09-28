"""
ULTRON Security — Loitering Detection Engine
============================================
Deterministic spatial-temporal behavior analytics.

A person is confirmed to be loitering if:
  1. They have been continuously present for >= LOITERING_THRESHOLD_S (e.g. 90s).
  2. Their cumulative spatial displacement remains confined within a localized
     radius (indicating lingering / casing the doorway rather than normal transit).

Emits LOITERING_DETECTED to the EventBus, escalating security state to SUSPICIOUS.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import config
from core.event_bus import EventBus, EventTypes
from vision.detector import PersonDetection


@dataclass
class PersonTrackHistory:
    """Historical spatial tracking data for a single person."""

    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    centroids: deque = field(default_factory=lambda: deque(maxlen=60))
    loitering_triggered: bool = False

    @property
    def dwell_time(self) -> float:
        return time.time() - self.first_seen

    def get_bounding_radius(self) -> float:
        """Calculate the maximum radial displacement from the starting position."""
        if len(self.centroids) < 2:
            return 0.0

        cx0, cy0 = self.centroids[0]
        max_dist = 0.0
        for cx, cy in self.centroids:
            dist = math.hypot(cx - cx0, cy - cy0)
            if dist > max_dist:
                max_dist = dist
        return max_dist


class LoiteringDetector:
    """
    Analyzes tracked persons across frames for stationary lingering.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self.threshold_s = getattr(config, "LOITERING_THRESHOLD_S", 90.0)
        self.radius_ratio = getattr(config, "LOITERING_RADIUS_RATIO", 0.20)

        self._tracks: dict[int, PersonTrackHistory] = {}
        self.event_bus.subscribe(EventTypes.PERSON_LEFT, self._on_person_left)

    def update_tracks(
        self, persons: list[PersonDetection], frame_width: int, frame_height: int
    ) -> list[int]:
        """
        Update tracking history and check for loitering violations.

        Returns:
            list of track_ids currently confirmed as loitering
        """
        if not getattr(config, "LOITERING_DETECTION_ENABLED", True):
            return []

        frame_diag = math.hypot(frame_width, frame_height)
        max_allowed_radius = frame_diag * self.radius_ratio
        now = time.time()
        loitering_ids = []

        for p in persons:
            if p.track_id < 0:
                continue

            tid = p.track_id
            if tid not in self._tracks:
                self._tracks[tid] = PersonTrackHistory(track_id=tid)

            th = self._tracks[tid]
            th.last_seen = now
            th.centroids.append(p.center)

            # Check if dwell threshold is exceeded
            if th.dwell_time >= self.threshold_s:
                # Check if spatial displacement is within stationary threshold
                disp = th.get_bounding_radius()
                if disp <= max_allowed_radius or len(th.centroids) < 5:
                    loitering_ids.append(tid)

                    # Fire event once per loitering session
                    if not th.loitering_triggered:
                        th.loitering_triggered = True
                        print(
                            f"[ULTRON Security] LOITERING DETECTED: Person ID:{tid} "
                            f"(dwell: {th.dwell_time:.1f}s, displacement: {disp:.1f}px)"
                        )
                        self.event_bus.publish(EventTypes.LOITERING_DETECTED, {
                            "track_id": tid,
                            "duration": th.dwell_time,
                            "displacement_px": disp,
                            "timestamp": now,
                        })

        return loitering_ids

    def _on_person_left(self, event):
        track_id = event.data.get("track_id", -1)
        self._tracks.pop(track_id, None)
