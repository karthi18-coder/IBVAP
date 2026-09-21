from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


class Detector:
    """
    IBVAP real object detector.

    Detects actual objects present in:
    - Uploaded images
    - Uploaded videos
    - OpenCV / NumPy frames
    - Webcam frames
    - RTSP frames

    No artificial/demo detections are generated.

    Relevant COCO classes:
        person
        car
        truck
        bus
        motorcycle
        bicycle
    """

    PERSON_CLASS = "person"

    VEHICLE_CLASSES = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
    }

    RELEVANT_CLASSES = {
        PERSON_CLASS,
        *VEHICLE_CLASSES,
    }

    CONFIDENCE_THRESHOLD = 0.35

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float | None = None,
    ) -> None:

        self.model = None
        self.mode = "YOLOv8"

        self.model_path = Path(model_path)
        self.confidence_threshold = (
            float(os.getenv("IBVAP_CONFIDENCE", str(self.CONFIDENCE_THRESHOLD)))
            if confidence is None
            else confidence
        )

        self._load_yolo()

    # ================================================================
    # LOAD MODEL
    # ================================================================

    def _load_yolo(self) -> None:

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed.\n"
                "Run: pip install ultralytics"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}\n"
                f"Place yolov8n.pt in the project root."
            )

        try:
            self.model = YOLO(str(self.model_path))

            print(
                f"[Detector] YOLOv8 loaded: "
                f"{self.model_path}"
            )

        except Exception as exc:
            raise RuntimeError(
                f"Could not load YOLO model: {exc}"
            ) from exc

    # ================================================================
    # PUBLIC DETECTION
    # ================================================================

    def detect(self, frame: Any) -> list[dict]:

        if frame is None:
            return []

        if self.model is None:
            raise RuntimeError(
                "YOLO model is not loaded."
            )

        return self._detect_yolo(frame)

    # ================================================================
    # REAL YOLO DETECTION
    # ================================================================

    def _normalize_frame(self, frame: Any) -> Any:
        if frame is None:
            return None

        if hasattr(frame, "image"):
            frame = frame.image

        if isinstance(frame, Image.Image):
            frame = np.asarray(frame.convert("RGB"))

        if isinstance(frame, np.ndarray):
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame

        return frame

    def _detect_yolo(self, frame: Any) -> list[dict]:

        try:
            frame = self._normalize_frame(frame)

            if frame is None:
                return []

            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
            )

            if not results:
                return []

            result = results[0]

            if result.boxes is None:
                return []

            names = result.names

            detections: list[dict] = []

            for index, box in enumerate(result.boxes):

                confidence = float(
                    box.conf[0].item()
                )

                if confidence < self.confidence_threshold:
                    continue

                class_id = int(
                    box.cls[0].item()
                )

                class_name = str(
                    names[class_id]
                ).lower()

                # Ignore irrelevant COCO classes
                if class_name not in self.RELEVANT_CLASSES:
                    continue

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                x1, y1, x2, y2 = [
                    max(0, int(round(value)))
                    for value in coordinates
                ]

                # Ignore invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

                detections.append(
                    {
                        "id": index + 1,

                        "class_id": class_id,

                        "class_name": class_name,

                        "confidence": round(
                            confidence,
                            3,
                        ),

                        "box": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],

                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],

                        "center": [
                            center_x,
                            center_y,
                        ],

                        "source": "YOLOv8",
                    }
                )

            return detections

        except Exception as exc:

            print(
                f"[Detector] YOLO detection error: {exc}"
            )

            # IMPORTANT:
            # Never create fake detections.
            raise

    # ================================================================
    # COUNTS
    # ================================================================

    def get_counts(
        self,
        detections: list[dict],
    ) -> dict:

        people = sum(
            1
            for item in detections
            if item.get("class_name") == "person"
        )

        vehicles = sum(
            1
            for item in detections
            if item.get("class_name")
            in self.VEHICLE_CLASSES
        )

        return {
            "total": len(detections),
            "people": people,
            "vehicles": vehicles,
        }

    # ================================================================
    # STATUS
    # ================================================================

    def get_status(self) -> dict:

        return {
            "mode": self.mode,
            "model": str(self.model_path),
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "demo_mode": False,
        }

    # ================================================================
    # RESET
    # ================================================================

    def reset(self) -> None:
        """
        Detector itself does not maintain tracking state.
        Tracker handles persistent IDs.
        """
        pass