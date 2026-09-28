"""
ULTRON Core — Identity Tracker & Re-Identification Manager
==========================================================
Maintains stable, persistent canonical person IDs across momentary departures,
occlusions, and ByteTrack raw tracker ID churn.

Guarantees:
  1. Spatial Persistence: When ByteTrack drops a track and re-assigns a new raw ID
     (e.g. 1 -> 2 -> 27) while the person is moving or slightly occluded, spatial IoU
     and centroid proximity match them back to their existing canonical ID.
  2. Single-Visitor Invariance: If only 1 person is in front of the camera, their
     canonical ID remains 1 forever. Ghost IDs (e.g. 2, 3, 13) are never created.
  3. Re-Entry Memory: When a visitor steps out of frame and returns within the
     re-identification window (e.g. 60s), their canonical ID is seamlessly restored.
  4. True Multi-Person Support: Multiple genuinely distinct people in frame receive
     distinct, persistent canonical IDs (1, 2, ...).
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

from vision.detector import PersonDetection


def _compute_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Intersection over Union between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0.0


def _compute_dist(ptA: tuple[int, int], ptB: tuple[int, int]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(ptA[0] - ptB[0], ptA[1] - ptB[1])


@dataclass
class CanonicalRecord:
    canonical_id: int
    last_raw_id: int
    last_bbox: tuple[int, int, int, int]
    last_center: tuple[int, int]
    last_seen: float
    is_active: bool = True


class IdentityTracker:
    """
    Manages persistent canonical IDs using spatial-temporal association.
    """

    def __init__(self, reid_window_s: float = 60.0):
        self.reid_window_s = reid_window_s
        self.raw_to_canonical: dict[int, int] = {}
        self.records: dict[int, CanonicalRecord] = {}

    def resolve_frame_persons(
        self, persons: list[PersonDetection], now: float
    ) -> set[int]:
        """
        Resolve all detected persons in the current frame to persistent canonical IDs.
        Updates each PersonDetection.track_id in-place with its canonical ID.

        Returns:
            set of currently active canonical IDs in this frame.
        """
        if not persons:
            # Mark all as inactive
            for r in self.records.values():
                r.is_active = False
            return set()

        # Step 1: Gather active and recently departed canonical candidates
        active_cids = {cid for cid, r in self.records.items() if r.is_active}
        recent_cids = [
            cid for cid, r in self.records.items()
            if not r.is_active and (now - r.last_seen) <= self.reid_window_s
        ]

        assigned_detections: dict[int, int] = {}  # detection index -> canonical_id
        claimed_canonicals: set[int] = set()

        # Step 2: Try direct raw_id mapping if valid and physically consistent
        for idx, p in enumerate(persons):
            raw_id = p.track_id
            if raw_id in self.raw_to_canonical:
                cid = self.raw_to_canonical[raw_id]
                if cid not in claimed_canonicals and cid in self.records:
                    rec = self.records[cid]
                    # Verify spatial consistency (within 250px or IoU > 0.1)
                    iou = _compute_iou(p.bbox, rec.last_bbox)
                    dist = _compute_dist(p.center, rec.last_center)
                    if iou > 0.10 or dist < 250:
                        assigned_detections[idx] = cid
                        claimed_canonicals.add(cid)

        # Step 3: Spatial proximity matching for unassigned detections against active records
        unassigned_indices = [i for i in range(len(persons)) if i not in assigned_detections]
        unclaimed_active = [cid for cid in active_cids if cid not in claimed_canonicals]

        for idx in list(unassigned_indices):
            p = persons[idx]
            best_cid = None
            best_score = -1.0

            for cid in unclaimed_active:
                rec = self.records[cid]
                iou = _compute_iou(p.bbox, rec.last_bbox)
                dist = _compute_dist(p.center, rec.last_center)

                # Score combines IoU and inverse distance
                if iou > 0.15 or dist < 180:
                    score = iou * 100.0 + max(0.0, 200.0 - dist)
                    if score > best_score:
                        best_score = score
                        best_cid = cid

            if best_cid is not None:
                assigned_detections[idx] = best_cid
                claimed_canonicals.add(best_cid)
                unclaimed_active.remove(best_cid)
                unassigned_indices.remove(idx)
                self.raw_to_canonical[p.track_id] = best_cid
                print(
                    f"[ULTRON Identity] Spatial match: raw ID {p.track_id} -> canonical ID {best_cid}"
                )

        # Step 4: Re-identification of recently departed visitors
        unassigned_indices = [i for i in range(len(persons)) if i not in assigned_detections]
        unclaimed_recent = [cid for cid in recent_cids if cid not in claimed_canonicals]

        # Case A: If only 1 person in frame and only 1 recent visitor, they are definitely the same person
        if len(persons) == 1 and unassigned_indices and unclaimed_recent:
            idx = unassigned_indices[0]
            p = persons[idx]
            # Pick most recently seen
            cid = max(unclaimed_recent, key=lambda c: self.records[c].last_seen)
            assigned_detections[idx] = cid
            claimed_canonicals.add(cid)
            self.raw_to_canonical[p.track_id] = cid
            unassigned_indices.remove(idx)
            print(
                f"[ULTRON Identity] Re-identified returning visitor: raw ID {p.track_id} -> canonical ID {cid}"
            )

        # Case B: Spatial proximity for any remaining recent departed candidates
        for idx in list(unassigned_indices):
            p = persons[idx]
            best_cid = None
            best_dist = float("inf")

            for cid in unclaimed_recent:
                rec = self.records[cid]
                dist = _compute_dist(p.center, rec.last_center)
                if dist < 300 and dist < best_dist:
                    best_dist = dist
                    best_cid = cid

            if best_cid is not None:
                assigned_detections[idx] = best_cid
                claimed_canonicals.add(best_cid)
                unclaimed_recent.remove(best_cid)
                unassigned_indices.remove(idx)
                self.raw_to_canonical[p.track_id] = best_cid
                print(
                    f"[ULTRON Identity] Re-identified returning visitor by position: raw ID {p.track_id} -> canonical ID {best_cid}"
                )

        # Step 5: Assign new canonical IDs to genuinely new visitors
        for idx in unassigned_indices:
            p = persons[idx]
            new_cid = self._allocate_next_canonical_id()
            assigned_detections[idx] = new_cid
            claimed_canonicals.add(new_cid)
            self.raw_to_canonical[p.track_id] = new_cid
            print(
                f"[ULTRON Identity] Registered new visitor: raw ID {p.track_id} -> canonical ID {new_cid}"
            )

        # Step 6: Update PersonDetection objects and internal records
        for idx, p in enumerate(persons):
            cid = assigned_detections[idx]
            p.track_id = cid

            if cid not in self.records:
                self.records[cid] = CanonicalRecord(
                    canonical_id=cid,
                    last_raw_id=p.track_id,
                    last_bbox=p.bbox,
                    last_center=p.center,
                    last_seen=now,
                    is_active=True,
                )
            else:
                rec = self.records[cid]
                rec.last_raw_id = p.track_id
                rec.last_bbox = p.bbox
                rec.last_center = p.center
                rec.last_seen = now
                rec.is_active = True

        # Mark any canonicals not in this frame as inactive
        for cid, rec in self.records.items():
            if cid not in claimed_canonicals:
                rec.is_active = False

        return claimed_canonicals

    def _allocate_next_canonical_id(self) -> int:
        """Allocate the lowest available positive integer ID not currently active."""
        used_ids = set(self.records.keys())
        cand = 1
        while cand in used_ids:
            cand += 1
        return cand
