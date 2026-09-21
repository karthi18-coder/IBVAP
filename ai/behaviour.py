from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BehaviourEngine:
    def __init__(self, evidence_dir: Path, events_file: Path) -> None:
        self.evidence_dir = evidence_dir
        self.events_file = events_file

        self.last_event_tick = -10
        self.tick = 0

        self.cooldowns: dict[str, int] = {}

        # Stores the previous position of each tracked object.
        self.previous_positions: dict[int, tuple[float, float]] = {}

        # Stores whether a tracked object was inside the restricted
        # zone during its previous frame.
        self.previous_zone_state: dict[int, bool] = {}

        self.event_history: list[dict[str, Any]] = []

        # Restricted-zone polygon.
        # Format:
        # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        self.fence_ratio = float(
            os.getenv("IBVAP_FENCE_RATIO", "0.35")
        )

        self.zone_points = self._load_zone_points()

        self.evidence_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # LOAD RESTRICTED ZONE
    # ========================================================

    def _load_zone_points(
        self,
    ) -> list[tuple[float, float]] | None:

        raw = os.getenv(
            "IBVAP_ZONE_POLYGON",
            "",
        ).strip()

        if not raw:
            return None

        try:
            parsed = json.loads(raw)

            if not isinstance(parsed, list):
                return None

            if len(parsed) < 3:
                return None

            points = []

            for item in parsed:

                if (
                    not isinstance(
                        item,
                        (list, tuple),
                    )
                    or len(item) != 2
                ):
                    return None

                points.append(
                    (
                        float(item[0]),
                        float(item[1]),
                    )
                )

            return points

        except Exception:
            return None

    # ========================================================
    # COOLDOWN
    # ========================================================

    def _cooldown_ok(
        self,
        key: str,
        frames: int = 28,
    ) -> bool:

        last_tick = self.cooldowns.get(
            key,
            -9999,
        )

        return (
            self.tick - last_tick
            >= frames
        )

    def _mark_cooldown(
        self,
        key: str,
    ) -> None:

        self.cooldowns[key] = self.tick

    # ========================================================
    # POINT INSIDE RESTRICTED ZONE
    # ========================================================

    def _track_in_zone(
        self,
        track: dict,
    ) -> bool:

        if not self.zone_points:
            return False

        center = track.get("center")

        if not center or len(center) != 2:
            return False

        x = float(center[0])
        y = float(center[1])

        inside = False

        n = len(self.zone_points)

        for i in range(n):

            x1, y1 = self.zone_points[i]

            x2, y2 = self.zone_points[
                (i + 1) % n
            ]

            intersects = (
                (y1 > y) != (y2 > y)
            ) and (
                x
                <
                (
                    (x2 - x1)
                    * (y - y1)
                    / (y2 - y1 + 1e-9)
                    + x1
                )
            )

            if intersects:
                inside = not inside

        return inside

    # ========================================================
    # MAIN BEHAVIOUR ANALYSIS
    # ========================================================

    def evaluate(
        self,
        tracks: list[dict],
        frame: Any,
    ) -> list[dict]:

        self.tick += 1

        events: list[dict] = []

        # ----------------------------------------------------
        # RESTRICTED ZONE CHECK
        # ----------------------------------------------------

        if self.zone_points:

            current_track_ids = set()

            for track in tracks:

                track_id = track.get(
                    "track_id"
                )

                if track_id is None:
                    continue

                current_track_ids.add(
                    track_id
                )

                # Current position
                center = track.get(
                    "center"
                )

                if not center or len(center) != 2:
                    continue

                current_position = (
                    float(center[0]),
                    float(center[1]),
                )

                # Is the object currently inside?
                currently_inside = (
                    self._track_in_zone(track)
                )

                # Was the object inside during
                # the previous frame?
                previously_inside = (
                    self.previous_zone_state.get(
                        track_id,
                        False,
                    )
                )

                # ------------------------------------------------
                # CROSSING DETECTED
                #
                # Outside -> Inside
                # ------------------------------------------------

                crossed_into_zone = (
                    not previously_inside
                    and currently_inside
                )

                if crossed_into_zone:

                    key = (
                        f"virtual_fence:"
                        f"{track_id}"
                    )

                    if self._cooldown_ok(
                        key,
                        28,
                    ):

                        event = self.create_event(
                            "Virtual Zone Violation",
                            [track],
                            frame,
                            track.get(
                                "confidence",
                                0.0,
                            ),
                        )

                        events.append(event)

                        self._mark_cooldown(
                            key
                        )

                # Save current state for
                # the next frame.
                self.previous_zone_state[
                    track_id
                ] = currently_inside

                self.previous_positions[
                    track_id
                ] = current_position

            # ------------------------------------------------
            # CLEAN OLD TRACK IDS
            # ------------------------------------------------

            old_ids = set(
                self.previous_zone_state.keys()
            ) - current_track_ids

            for old_id in old_ids:

                self.previous_zone_state.pop(
                    old_id,
                    None,
                )

                self.previous_positions.pop(
                    old_id,
                    None,
                )

        return events

    # ========================================================
    # FRAME SIZE
    # ========================================================

    @staticmethod
    def _frame_size(
        frame: Any,
    ) -> tuple[int, int]:

        if isinstance(
            getattr(frame, "size", None),
            tuple,
        ):

            width, height = frame.size

            return (
                int(width),
                int(height),
            )

        shape = getattr(
            frame,
            "shape",
            (0, 0),
        )

        return (
            int(shape[1])
            if len(shape) > 1
            else 0,

            int(shape[0])
            if shape
            else 0,
        )

    # ========================================================
    # CREATE EVENT
    # ========================================================

    def create_event(
        self,
        event_type: str,
        tracks: list[dict],
        frame: Any,
        confidence: float = 0.0,
    ) -> dict:

        timestamp = datetime.now(
            timezone.utc
        )

        stamp = timestamp.strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]

        safe_type = "_".join(
            event_type.lower().split()
        )

        event_id = (
            f"EV-{stamp}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        filename = (
            f"event_{stamp}_"
            f"{safe_type}.jpg"
        )

        path = (
            self.evidence_dir
            / filename
        )

        evidence_saved = False

        try:

            if (
                frame is not None
                and hasattr(frame, "save")
            ):

                frame.save(
                    path,
                    "JPEG",
                )

                evidence_saved = True

            elif frame is not None:

                import cv2

                evidence_saved = bool(
                    cv2.imwrite(
                        str(path),
                        frame,
                    )
                )

        except (
            ImportError,
            OSError,
            ValueError,
        ):

            evidence_saved = False

        # ----------------------------------------------------
        # EVENT SEVERITY
        # ----------------------------------------------------

        severity = "MEDIUM"

        if (
            "violation"
            in event_type.lower()
        ):

            severity = "HIGH"

        # ----------------------------------------------------
        # EVENT DATA
        # ----------------------------------------------------

        event = {
            "event_id": event_id,

            "event_type": event_type,

            "timestamp":
                timestamp.isoformat(),

            "camera_id":
                "CCTV-01",

            "severity":
                severity,

            "status":
                "NEW",

            "tracking_id":
                (
                    tracks[0]["track_id"]
                    if tracks
                    else None
                ),

            "confidence":
                round(
                    confidence,
                    2,
                ),

            "evidence":
                (
                    filename
                    if evidence_saved
                    else None
                ),

            "evidence_path":
                (
                    str(path)
                    if evidence_saved
                    else None
                ),
        }

        self.event_history.insert(
            0,
            event,
        )

        self.event_history = (
            self.event_history[:50]
        )

        return event

    # ========================================================
    # EVENT HISTORY
    # ========================================================

    def history(
        self,
    ) -> list[dict]:

        return list(
            self.event_history
        )