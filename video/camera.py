from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from io import BytesIO

from PIL import Image, ImageDraw


class CameraSource:
    def __init__(self, source: str | int | None = None) -> None:
        self.source = None
        self.source_label = "IDLE"
        self.capture = None
        self.tick = 0
        self._demo_mode = False
        self.latest_frame = None
        self._frame_lock = threading.Lock()
        self._capture_thread = None
        self._capture_stop = threading.Event()
        self._read_failures = 0
        self._is_file_source = False
        self._capture_finished = threading.Event()
        self._frame_interval = 0.0
        if source is not None:
            self.set_source(source)

    @staticmethod
    def _normalize_source(source: str | int | None) -> str | int | None:
        if source is None:
            return None
        if isinstance(source, int):
            return source
        value = str(source).strip()
        if not value:
            return None
        if value.lower() in {"demo", "default", "local", "simulated"}:
            return None
        if value.lower() == "webcam":
            return 0
        return value

    def _configure_capture(self) -> None:
        if self.capture is None:
            return
        try:
            import cv2
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self._is_file_source:
                self.capture.set(cv2.CAP_PROP_FPS, 30)
            if self._is_file_source:
                fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
                self._frame_interval = 1.0 / fps if fps > 1.0 else 0.0
        except Exception:
            pass

    def set_source(self, source: str | int | None) -> bool:
        self.release("switching camera source")
        self.source = self._normalize_source(source)
        self._demo_mode = False
        self._is_file_source = isinstance(self.source, str) and Path(self.source).exists()
        self._capture_finished.clear()
        self._frame_interval = 0.0
        self.source_label = "IDLE" if self.source is None else str(self.source)

        if self.source is None:
            return True

        try:
            import cv2
        except ImportError:
            self.source = None
            self.source_label = "ERROR: OpenCV unavailable"
            return False

        if isinstance(self.source, int):
            self.capture = cv2.VideoCapture(self.source)
        else:
            path = Path(self.source)
            if path.exists():
                self.capture = cv2.VideoCapture(str(path))
            else:
                self.capture = cv2.VideoCapture(self.source)

        if isinstance(self.source, int):
            print("[CAMERA] Opening webcam...")
        elif self._is_file_source:
            print(f"[VIDEO] Opening CCTV video: {Path(self.source).name}")
        else:
            print(f"[CAMERA] Opening source: {self.source}")
        if not self.capture.isOpened():
            print(f"[CAMERA] Could not open source: {self.source}")
            self.release("camera failed to open")
            self.source = None
            self.source_label = "ERROR: source could not be opened"
            return False

        if isinstance(self.source, int):
            print("[CAMERA] Webcam opened successfully")
        elif self._is_file_source:
            print("[VIDEO] CCTV video opened successfully")
        else:
            print("[CAMERA] Source opened successfully")
        self._configure_capture()
        self.start_capture()
        return True

    def start_capture(self) -> None:
        if self.capture is None:
            return
        self._capture_stop.clear()
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print("[VIDEO] Video capture thread started" if self._is_file_source else "[CAMERA] Capture thread started")

    def stop_capture(self) -> None:
        self._capture_stop.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.5)
        self._capture_thread = None

    def _capture_loop(self) -> None:
        while not self._capture_stop.is_set() and self.capture is not None:
            try:
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    if self._is_file_source:
                        self._capture_finished.set()
                        print("[VIDEO] CCTV video finished")
                        break
                    self._read_failures += 1
                    if self._read_failures < 30:
                        if self._read_failures == 1 or self._read_failures % 10 == 0:
                            print(f"[CAMERA] Read failure ({self._read_failures}/30)")
                        continue
                    print("[CAMERA] Too many consecutive read failures")
                    self.source_label = "ERROR: camera read failed"
                    break
                with self._frame_lock:
                    self.latest_frame = frame
                self._read_failures = 0
                self.tick += 1
                if self._is_file_source and self._frame_interval:
                    self._capture_stop.wait(self._frame_interval)
            except Exception:
                if self._is_file_source:
                    self._capture_finished.set()
                    break
                self._read_failures += 1
                if self._read_failures >= 30:
                    print("[CAMERA] Too many consecutive read failures")
                    self.source_label = "ERROR: camera read failed"
                    break
                continue

    def release(self, reason: str = "unspecified") -> None:
        print("[DEBUG] camera.release() CALLED")
        print(f"[DEBUG] Releasing camera because: {reason}")
        print(f"[DEBUG] current thread: {threading.current_thread().name}")
        self.stop_capture()
        if self.capture is not None:
            was_file_source = self._is_file_source
            self.capture.release()
            print("[VIDEO] CCTV video released" if was_file_source else "[CAMERA] Webcam released")
        self.capture = None
        self._is_file_source = False
        self._frame_interval = 0.0
        with self._frame_lock:
            self.latest_frame = None

    @property
    def is_file_source(self) -> bool:
        return self._is_file_source

    @property
    def capture_finished(self) -> bool:
        return self._capture_finished.is_set()

    @staticmethod
    def load_image(path: str | Path) -> Any:
        return Image.open(path).convert("RGB")

    def read(self) -> Any:
        with self._frame_lock:
            frame = self.latest_frame
        if frame is None:
            return None
        return frame.copy()

    def read_latest(self) -> Any:
        with self._frame_lock:
            frame = self.latest_frame
        if frame is None:
            return None
        return frame.copy()


def frame_to_jpeg(frame: Any, tracks: list[dict]) -> bytes:
    if hasattr(frame, "image") or isinstance(frame, Image.Image):
        image = frame.image.copy() if hasattr(frame, "image") else frame.copy()
        draw = ImageDraw.Draw(image)
        for track in tracks:
            x1, y1, x2, y2 = track["box"]
            draw.rectangle((x1, y1, x2, y2), outline="#efb45f", width=3)
            identity = track.get("person_id") or track.get("track_id")
            draw.text((x1, max(0, y1 - 18)), f"{track['class_name']} #{identity}", fill="#efb45f")
        output = BytesIO()
        image.save(output, "JPEG")
        return output.getvalue()
    import cv2
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (95, 180, 239), 2)
        identity = track.get("person_id") or track.get("track_id")
        cv2.putText(
            annotated,
            f"{track.get('class_name', 'object')} #{identity}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (95, 180, 239),
            2,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, 70],
    )
    return encoded.tobytes() if ok else b""
