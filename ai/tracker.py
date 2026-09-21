from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ============================================================
# PERSISTENT PERSON IDENTITY
# ============================================================

@dataclass
class PersonIdentity:
    person_id: int

    # Current averaged appearance feature
    feature: np.ndarray

    # Additional appearance samples
    feature_samples: list[np.ndarray] = field(default_factory=list)

    # Last known position
    last_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    # Last time detected
    last_seen: float = 0.0

    # Currently visible
    active: bool = True

    # Number of successful matches
    match_count: int = 0


# ============================================================
# TRACKER
# ============================================================

class Tracker:
    """
    Multi-object tracker with persistent person identity.

    track_id:
        Temporary frame-to-frame tracking ID.

    person_id:
        Persistent identity ID for a person during the
        current monitoring session.

    The tracker combines:

        1. IoU
        2. Centre distance
        3. HSV appearance
        4. Colour distribution
        5. Bounding-box shape
        6. Multiple appearance samples
        7. Persistent identity history
    """

    def __init__(
        self,
        max_missing_seconds: float = 30.0,
        appearance_threshold: float = 0.52,
        distance_threshold: float = 300.0,
        track_max_missing_seconds: float = 2.5,
    ):

        # ----------------------------------------------------
        # SETTINGS
        # ----------------------------------------------------

        self.max_missing_seconds = max_missing_seconds

        self.appearance_threshold = appearance_threshold

        self.distance_threshold = distance_threshold

        self.track_max_missing_seconds = (
            track_max_missing_seconds
        )

        # Maximum appearance samples stored per person
        self.max_feature_samples = 8

        # ----------------------------------------------------
        # ID COUNTERS
        # ----------------------------------------------------

        self.next_track_id = 1
        self.next_person_id = 1

        self._update_count = 0

        # ----------------------------------------------------
        # TEMPORARY TRACKS
        # ----------------------------------------------------

        self.active_tracks: dict[int, dict[str, Any]] = {}

        # ----------------------------------------------------
        # PERSISTENT PERSON IDENTITIES
        # ----------------------------------------------------

        self.identities: dict[int, PersonIdentity] = {}

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """
        Completely reset the current monitoring session.
        """

        self.active_tracks.clear()
        self.identities.clear()

        self.next_track_id = 1
        self.next_person_id = 1

        self._update_count = 0

    # ========================================================
    # BOUNDING BOX HELPERS
    # ========================================================

    def _clamp_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))

        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))

        return x1, y1, x2, y2

    # ========================================================
    # CENTRE DISTANCE
    # ========================================================

    def _centre_distance(
        self,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:

        ax = (bbox_a[0] + bbox_a[2]) / 2.0
        ay = (bbox_a[1] + bbox_a[3]) / 2.0

        bx = (bbox_b[0] + bbox_b[2]) / 2.0
        by = (bbox_b[1] + bbox_b[3]) / 2.0

        return float(
            np.sqrt(
                (ax - bx) ** 2
                +
                (ay - by) ** 2
            )
        )

    # ========================================================
    # IOU
    # ========================================================

    def _iou(
        self,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:

        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        intersection = iw * ih

        area_a = (
            max(0, ax2 - ax1)
            *
            max(0, ay2 - ay1)
        )

        area_b = (
            max(0, bx2 - bx1)
            *
            max(0, by2 - by1)
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return float(
            intersection / union
        )

    # ========================================================
    # BBOX SHAPE SIMILARITY
    # ========================================================

    def _shape_similarity(
        self,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:

        aw = max(
            1,
            bbox_a[2] - bbox_a[0],
        )

        ah = max(
            1,
            bbox_a[3] - bbox_a[1],
        )

        bw = max(
            1,
            bbox_b[2] - bbox_b[0],
        )

        bh = max(
            1,
            bbox_b[3] - bbox_b[1],
        )

        ratio_a = aw / ah
        ratio_b = bw / bh

        difference = abs(
            ratio_a - ratio_b
        )

        return max(
            0.0,
            1.0 - min(
                difference / 0.8,
                1.0,
            ),
        )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def _extract_feature(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:

        if frame is None:
            return None

        try:

            bbox = self._clamp_bbox(
                frame,
                bbox,
            )

            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[
                y1:y2,
                x1:x2,
            ]

            if crop.size == 0:
                return None

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            crop = cv2.resize(
                crop,
                (64, 128),
                interpolation=cv2.INTER_AREA,
            )

            # ------------------------------------------------
            # Slight blur to reduce camera noise
            # ------------------------------------------------

            crop = cv2.GaussianBlur(
                crop,
                (3, 3),
                0,
            )

            # ------------------------------------------------
            # HSV
            # ------------------------------------------------

            hsv = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2HSV,
            )

            # ------------------------------------------------
            # Global HSV histogram
            # ------------------------------------------------

            hist_global = cv2.calcHist(
                [hsv],
                [0, 1],
                None,
                [24, 24],
                [0, 180, 0, 256],
            )

            cv2.normalize(
                hist_global,
                hist_global,
            )

            # ------------------------------------------------
            # Split person into 3 vertical regions
            #
            # Head / upper body
            # Middle body
            # Lower body
            #
            # This helps distinguish two people with
            # different clothing.
            # ------------------------------------------------

            height = crop.shape[0]

            regions = [
                crop[
                    0:int(height * 0.33)
                ],
                crop[
                    int(height * 0.33):
                    int(height * 0.70)
                ],
                crop[
                    int(height * 0.70):
                ],
            ]

            region_features = []

            for region in regions:

                if region.size == 0:
                    continue

                region_hsv = cv2.cvtColor(
                    region,
                    cv2.COLOR_BGR2HSV,
                )

                hist = cv2.calcHist(
                    [region_hsv],
                    [0, 1],
                    None,
                    [16, 16],
                    [0, 180, 0, 256],
                )

                cv2.normalize(
                    hist,
                    hist,
                )

                region_features.append(
                    hist.flatten()
                )

            # ------------------------------------------------
            # HSV feature
            # ------------------------------------------------

            feature_parts = [
                hist_global.flatten()
            ]

            feature_parts.extend(
                region_features
            )

            # ------------------------------------------------
            # Colour statistics
            # ------------------------------------------------

            mean_hsv = np.mean(
                hsv,
                axis=(0, 1),
            )

            std_hsv = np.std(
                hsv,
                axis=(0, 1),
            )

            feature_parts.append(
                mean_hsv.astype(
                    np.float32
                )
                /
                np.array(
                    [180.0, 255.0, 255.0],
                    dtype=np.float32,
                )
            )

            feature_parts.append(
                std_hsv.astype(
                    np.float32
                )
                /
                np.array(
                    [180.0, 255.0, 255.0],
                    dtype=np.float32,
                )
            )

            feature = np.concatenate(
                feature_parts
            ).astype(
                np.float32
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            norm = np.linalg.norm(
                feature
            )

            if norm > 0:
                feature /= norm

            return feature

        except Exception as exc:

            print(
                f"[TRACKER] Feature extraction error: {exc}"
            )

            return None

    # ========================================================
    # APPEARANCE SIMILARITY
    # ========================================================

    def _appearance_similarity(
        self,
        feature_a: np.ndarray | None,
        feature_b: np.ndarray | None,
    ) -> float:

        if feature_a is None:
            return 0.0

        if feature_b is None:
            return 0.0

        if feature_a.shape != feature_b.shape:
            return 0.0

        try:

            similarity = float(
                np.dot(
                    feature_a,
                    feature_b,
                )
            )

            return max(
                0.0,
                min(
                    1.0,
                    similarity,
                ),
            )

        except Exception:

            return 0.0

    # ========================================================
    # MULTI-SAMPLE APPEARANCE MATCH
    # ========================================================

    def _identity_similarity(
        self,
        feature: np.ndarray | None,
        identity: PersonIdentity,
    ) -> float:

        if feature is None:
            return 0.0

        scores = []

        # ----------------------------------------------------
        # Current averaged feature
        # ----------------------------------------------------

        current_score = (
            self._appearance_similarity(
                feature,
                identity.feature,
            )
        )

        scores.append(
            current_score
        )

        # ----------------------------------------------------
        # Historical appearance samples
        # ----------------------------------------------------

        for sample in identity.feature_samples:

            score = (
                self._appearance_similarity(
                    feature,
                    sample,
                )
            )

            scores.append(
                score
            )

        if not scores:
            return 0.0

        # Use strongest historical match.
        #
        # This is important because the current appearance
        # may differ from the first frame due to lighting,
        # movement or camera angle.
        return float(
            max(scores)
        )

    # ========================================================
    # FIND EXISTING PERSON
    # ========================================================

    def _find_existing_person(
        self,
        feature: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        now: float,
        excluded_person_ids: set[int] | None = None,
    ) -> int | None:

        if feature is None:
            return None

        if excluded_person_ids is None:
            excluded_person_ids = set()

        best_person_id = None
        best_score = -1.0

        # ----------------------------------------------------
        # Examine all known identities
        # ----------------------------------------------------

        for (
            person_id,
            identity,
        ) in self.identities.items():

            if person_id in excluded_person_ids:
                continue

            # ------------------------------------------------
            # Ignore very old identities
            # ------------------------------------------------

            if (
                now
                -
                identity.last_seen
                >
                self.max_missing_seconds
            ):
                continue

            # ------------------------------------------------
            # Appearance
            # ------------------------------------------------

            appearance = (
                self._identity_similarity(
                    feature,
                    identity,
                )
            )

            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            distance = (
                self._centre_distance(
                    bbox,
                    identity.last_bbox,
                )
            )

            distance_score = (
                1.0
                -
                min(
                    distance
                    /
                    self.distance_threshold,
                    1.0,
                )
            )

            # ------------------------------------------------
            # Shape
            # ------------------------------------------------

            shape_score = (
                self._shape_similarity(
                    bbox,
                    identity.last_bbox,
                )
            )

            # ------------------------------------------------
            # Different matching behaviour depending on
            # whether the person is close or has returned
            # from somewhere else.
            # ------------------------------------------------

            if distance <= self.distance_threshold:

                score = (
                    appearance * 0.65
                    +
                    distance_score * 0.20
                    +
                    shape_score * 0.15
                )

            else:

                # When person moved far away,
                # appearance becomes much more important.
                score = (
                    appearance * 0.85
                    +
                    shape_score * 0.15
                )

            # ------------------------------------------------
            # Require minimum appearance
            # ------------------------------------------------

            if appearance < self.appearance_threshold:
                continue

            if score > best_score:

                best_score = score
                best_person_id = person_id

        return best_person_id

    # ========================================================
    # CREATE PERSON IDENTITY
    # ========================================================

    def _create_person_identity(
        self,
        feature: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        now: float,
    ) -> int:

        person_id = self.next_person_id

        self.next_person_id += 1

        if feature is None:

            # Feature size used by current extractor.
            feature = np.zeros(
                1252,
                dtype=np.float32,
            )

        self.identities[
            person_id
        ] = PersonIdentity(

            person_id=person_id,

            feature=feature.copy(),

            feature_samples=[
                feature.copy()
            ],

            last_bbox=bbox,

            last_seen=now,

            active=True,

            match_count=1,
        )

        return person_id

    # ========================================================
    # UPDATE PERSON IDENTITY
    # ========================================================

    def _update_identity(
        self,
        person_id: int,
        feature: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        now: float,
    ):

        identity = self.identities.get(
            person_id
        )

        if identity is None:
            return

        # ----------------------------------------------------
        # Update averaged feature
        # ----------------------------------------------------

        if feature is not None:

            identity.feature = (
                identity.feature * 0.80
                +
                feature * 0.20
            )

            norm = np.linalg.norm(
                identity.feature
            )

            if norm > 0:
                identity.feature /= norm

            # ------------------------------------------------
            # Store historical sample
            # ------------------------------------------------

            identity.feature_samples.append(
                feature.copy()
            )

            if (
                len(
                    identity.feature_samples
                )
                >
                self.max_feature_samples
            ):

                identity.feature_samples.pop(
                    0
                )

        # ----------------------------------------------------
        # Update position
        # ----------------------------------------------------

        identity.last_bbox = bbox

        identity.last_seen = now

        identity.active = True

        identity.match_count += 1

    # ========================================================
    # FIND TEMPORARY TRACK
    # ========================================================

    def _find_best_track(
        self,
        bbox: tuple[int, int, int, int],
        class_id: int,
        feature: np.ndarray | None,
        matched_track_ids: set[int],
    ) -> int | None:

        best_track_id = None
        best_score = -1.0

        for (
            track_id,
            track,
        ) in self.active_tracks.items():

            if track_id in matched_track_ids:
                continue

            if track["class_id"] != class_id:
                continue

            old_bbox = track["bbox"]

            distance = (
                self._centre_distance(
                    bbox,
                    old_bbox,
                )
            )

            iou = self._iou(
                bbox,
                old_bbox,
            )

            # ------------------------------------------------
            # Appearance of temporary track
            # ------------------------------------------------

            track_feature = track.get(
                "feature"
            )

            appearance = (
                self._appearance_similarity(
                    feature,
                    track_feature,
                )
                if feature is not None
                and track_feature is not None
                else 0.0
            )

            # ------------------------------------------------
            # Reject impossible movement
            # ------------------------------------------------

            if (
                distance
                >
                self.distance_threshold
                and
                iou < 0.01
                and
                appearance
                <
                self.appearance_threshold
            ):
                continue

            distance_score = (
                1.0
                -
                min(
                    distance
                    /
                    self.distance_threshold,
                    1.0,
                )
            )

            # ------------------------------------------------
            # Track score
            # ------------------------------------------------

            score = (
                iou * 0.40
                +
                distance_score * 0.25
                +
                appearance * 0.35
            )

            if score > best_score:

                best_score = score
                best_track_id = track_id

        return best_track_id

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        now = time.time()

        self._update_count += 1

        results: list[dict[str, Any]] = []

        matched_track_ids: set[int] = set()

        matched_person_ids: set[int] = set()

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        for detection in detections:

            bbox_raw = (
                detection.get("bbox")
                or
                detection.get("box")
            )

            if (
                not bbox_raw
                or
                len(bbox_raw) != 4
            ):
                continue

            bbox = tuple(
                int(v)
                for v in bbox_raw
            )

            confidence = float(
                detection.get(
                    "confidence",
                    0.0,
                )
            )

            class_id = int(
                detection.get(
                    "class_id",
                    0,
                )
            )

            class_name = str(
                detection.get(
                    "class_name",
                    "unknown",
                )
            )

            is_person_detection = (
                class_name.lower()
                ==
                "person"
                or
                class_id == 0
            )

            # =================================================
            # PERSON
            # =================================================

            if is_person_detection:

                feature = (
                    self._extract_feature(
                        frame,
                        bbox,
                    )
                )

                # ------------------------------------------------
                # Step 1: Match temporary track
                # ------------------------------------------------

                track_id = (
                    self._find_best_track(
                        bbox,
                        class_id,
                        feature,
                        matched_track_ids,
                    )
                )

                if track_id is None:

                    track_id = (
                        self.next_track_id
                    )

                    self.next_track_id += 1

                matched_track_ids.add(
                    track_id
                )

                # ------------------------------------------------
                # Step 2: Check whether temporary track already
                # knows the persistent person
                # ------------------------------------------------

                person_id = None

                old_track = (
                    self.active_tracks.get(
                        track_id
                    )
                )

                if old_track is not None:

                    person_id = old_track.get(
                        "person_id"
                    )

                # ------------------------------------------------
                # Step 3: If no known person, use Re-ID
                # ------------------------------------------------

                if person_id is None:

                    person_id = (
                        self._find_existing_person(
                            feature,
                            bbox,
                            now,
                            matched_person_ids,
                        )
                    )

                # ------------------------------------------------
                # Step 4: Prevent duplicate ID in same frame
                # ------------------------------------------------

                if person_id in matched_person_ids:

                    person_id = None

                # ------------------------------------------------
                # Step 5: Create new persistent identity
                # ------------------------------------------------

                if person_id is None:

                    person_id = (
                        self._create_person_identity(
                            feature,
                            bbox,
                            now,
                        )
                    )

                else:

                    self._update_identity(
                        person_id,
                        feature,
                        bbox,
                        now,
                    )

                matched_person_ids.add(
                    person_id
                )

                # ------------------------------------------------
                # Save temporary track
                # ------------------------------------------------

                self.active_tracks[
                    track_id
                ] = {

                    "bbox": bbox,

                    "class_id": class_id,

                    "class_name": class_name,

                    "person_id": person_id,

                    "feature": feature,

                    "last_seen": now,

                }

                # ------------------------------------------------
                # Return result
                # ------------------------------------------------

                result = dict(
                    detection
                )

                result["track_id"] = (
                    track_id
                )

                result["person_id"] = (
                    person_id
                )

                results.append(
                    result
                )

            # =================================================
            # NON-PERSON OBJECT
            # =================================================

            else:

                track_id = (
                    self._find_best_track(
                        bbox,
                        class_id,
                        None,
                        matched_track_ids,
                    )
                )

                if track_id is None:

                    track_id = (
                        self.next_track_id
                    )

                    self.next_track_id += 1

                matched_track_ids.add(
                    track_id
                )

                self.active_tracks[
                    track_id
                ] = {

                    "bbox": bbox,

                    "class_id": class_id,

                    "class_name": class_name,

                    "person_id": None,

                    "feature": None,

                    "last_seen": now,

                }

                result = dict(
                    detection
                )

                result["track_id"] = (
                    track_id
                )

                result["person_id"] = None

                results.append(
                    result
                )

        # ========================================================
        # MARK IDENTITIES AS ACTIVE / INACTIVE
        # ========================================================

        for (
            person_id,
            identity,
        ) in self.identities.items():

            if (
                now
                -
                identity.last_seen
                >
                0.7
            ):

                identity.active = False

        # ========================================================
        # REMOVE OLD TEMPORARY TRACKS
        # ========================================================

        expired_tracks = []

        for (
            track_id,
            track,
        ) in self.active_tracks.items():

            if (
                now
                -
                track["last_seen"]
                >
                self.track_max_missing_seconds
            ):

                expired_tracks.append(
                    track_id
                )

        for track_id in expired_tracks:

            del self.active_tracks[
                track_id
            ]

        # DEBUG OUTPUT
        # ========================================================

        if self._update_count % 30 == 0:

            # IDs assigned to persons in the current frame
            assigned_ids = [
                item.get("person_id")
                for item in results
                if item.get("person_id") is not None
            ]

            # Track IDs in the current frame
            track_ids = [
                item.get("track_id")
                for item in results
                if item.get("track_id") is not None
            ]

            # Remove duplicates while preserving order
            assigned_ids = list(dict.fromkeys(assigned_ids))
            track_ids = list(dict.fromkeys(track_ids))

            print(
                "[TRACKER] "
                f"detections={len(detections)} | "
                f"active_tracks={sorted(self.active_tracks.keys())} | "
                f"tracks={track_ids} | "
                f"person_ids={assigned_ids}"
            )

            assigned_ids = [
                item.get(
                    "person_id"
                )
                for item in results
                if item.get(
                    "person_id"
                ) is not None
            ]

            track_ids = [
                item.get(
                    "track_id"
                )
                for item in results
            ]

            print(
                "[TRACKER] "
                f"detections={len(detections)} | "
                f"active_tracks={sorted(self.active_tracks)} | "
                f"track_ids={track_ids} | "
                f"person_ids={assigned_ids} | "
                f"identities={sorted(self.identities)}"
            )

        return results