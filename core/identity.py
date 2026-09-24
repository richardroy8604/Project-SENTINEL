"""
ULTRON Core — Identity Tracker & Re-Identification Manager
==========================================================
Maintains persistent canonical person IDs across momentary departures, occlusions,
and ByteTrack ID re-assignments.

Problem Solved:
  When a person steps out of the camera view and returns a few seconds or a minute later,
  ByteTrack assigns a new track ID (e.g. 1 -> 2). Without identity persistence,
  the system treats the returning visitor as a completely new person, increments
  person counts falsely, and spams greetings saying "another one arrived".

Solution:
  Maintains a memory of recently departed canonical IDs (up to 60s).
  When a new raw track ID appears while a previous person recently left and is not
  currently in frame, it re-maps the new track ID to the existing canonical ID.
"""

import time


class IdentityTracker:
    """
    Manages persistent canonical IDs across tracker re-assignments.
    """

    def __init__(self, reid_window_s: float = 60.0):
        self.reid_window_s = reid_window_s
        self.raw_to_canonical: dict[int, int] = {}
        self.departed_timestamps: dict[int, float] = {}  # canonical_id -> departure timestamp
        self.active_canonical_ids: set[int] = set()

    def resolve_track_id(self, raw_id: int, now: float) -> int:
        """Resolve a raw tracker ID to a stable canonical ID."""
        if raw_id < 0:
            return raw_id

        # 1. If already mapped, return it
        if raw_id in self.raw_to_canonical:
            canonical_id = self.raw_to_canonical[raw_id]
            self.departed_timestamps.pop(canonical_id, None)
            return canonical_id

        # 2. Check if a previously seen person departed recently and is not currently active
        candidate_id = None
        latest_time = 0.0
        for cid, dep_time in self.departed_timestamps.items():
            if cid not in self.active_canonical_ids and (now - dep_time) <= self.reid_window_s:
                if dep_time > latest_time:
                    latest_time = dep_time
                    candidate_id = cid

        if candidate_id is not None:
            # Re-associate to the recently departed person!
            self.raw_to_canonical[raw_id] = candidate_id
            self.departed_timestamps.pop(candidate_id, None)
            print(f"[ULTRON Identity] Re-identified returning visitor: raw ID {raw_id} -> canonical ID {candidate_id}")
            return candidate_id

        # 3. Genuinely new person
        self.raw_to_canonical[raw_id] = raw_id
        return raw_id

    def on_person_departed(self, canonical_id: int, now: float):
        """Record that a canonical person has departed."""
        self.departed_timestamps[canonical_id] = now
        self.active_canonical_ids.discard(canonical_id)

    def set_active_ids(self, active_canonicals: set[int]):
        """Update the set of currently visible canonical IDs."""
        self.active_canonical_ids = set(active_canonicals)
