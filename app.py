from __future__ import annotations
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename
from ai.behaviour import BehaviourEngine
from ai.detector import Detector
from ai.tracker import Tracker
from alert_manager import AlertManager
from database import EventDatabase
from video.camera import CameraSource, frame_to_jpeg


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video" / "samples" / "videos"
UPLOAD_DIR = BASE_DIR / "video" / "uploads"

EVIDENCE_DIR = BASE_DIR / "evidence"
EVENTS_FILE = BASE_DIR / "events.json"
DATABASE_PATH = BASE_DIR / "ibvap.db"

for directory in (
    EVIDENCE_DIR,
    VIDEO_DIR,
    UPLOAD_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGES = {"jpg", "jpeg", "png", "webp"}


# ============================================================
# DEFAULT VIDEO SOURCE
# ============================================================

VIDEO_CANDIDATES = [
    VIDEO_DIR / "cctv_test.mp4",
    VIDEO_DIR / "cctv_test.mp4.mp4",
    VIDEO_DIR / "sample.mp4",
    BASE_DIR / "videos" / "cctv_test.mp4",
]

VIDEO_FILE = next(
    (path for path in VIDEO_CANDIDATES if path.exists()),
    VIDEO_CANDIDATES[0],
)

video_source = None


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024


# ============================================================
# AI COMPONENTS
# ============================================================

camera = CameraSource()

detector = Detector()

tracker = Tracker()

database = EventDatabase(DATABASE_PATH)

behaviour = BehaviourEngine(
    EVIDENCE_DIR,
    EVENTS_FILE,
)

alert_manager = AlertManager(database)


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()

latest = {
    "frame": None,
    "detections": [],
    "events": [],
    "updated_at": None,
}

video_state = {
    "running": False,
    "status": "IDLE",
    "fps": 0.0,
    "source": "IDLE",
    "video_filename": None,
    "video_finished": False,
}

worker_lock = threading.Lock()
camera_io_lock = threading.Lock()
worker_stop = threading.Event()
worker_thread = None

frame_skip = max(
    1,
    int(os.getenv("IBVAP_FRAME_SKIP", "1")),
)
MOTION_THRESHOLD = float(os.getenv("IBVAP_MOTION_THRESHOLD", "18.0"))
AI_DETECTION_INTERVAL = float(os.getenv("IBVAP_AI_INTERVAL", "0.18"))
FPS_STATE = {
    "last_time": time.monotonic(),
    "frames": 0,
    "fps": 0.0,
}
AI_LAST_RUN = {
    "ts": 0.0,
}


def _motion_score(frame) -> float:
    try:
        import cv2
    except ImportError:
        return 100.0

    if frame is None:
        return 0.0

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        previous = getattr(_motion_score, "previous_frame", None)
        if previous is None:
            _motion_score.previous_frame = blurred
            return 0.0
        diff = cv2.absdiff(previous, blurred)
        score = float(cv2.mean(diff)[0])
        _motion_score.previous_frame = blurred
        return score
    except Exception:
        return 100.0


# ============================================================
# OBJECT CLASSIFICATION
# ============================================================

PERSON_CLASSES = {
    "person",
}

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


def is_person(item: dict) -> bool:
    return item.get("class_name", "").lower() in PERSON_CLASSES


def is_vehicle(item: dict) -> bool:
    return item.get("class_name", "").lower() in VEHICLE_CLASSES


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(frame=None) -> bool:
    """Read the newest frame and only run AI when the interval allows it."""

    if frame is None:
        with camera_io_lock:
            frame = camera.read_latest()

    if frame is None:
        return False
    FPS_STATE["frames"] += 1

    now_fps = time.monotonic()

    if now_fps - FPS_STATE["last_time"] >= 1.0:

        FPS_STATE["fps"] = (
            FPS_STATE["frames"]
            / (now_fps - FPS_STATE["last_time"])
        )

        FPS_STATE["frames"] = 0
        FPS_STATE["last_time"] = now_fps

        with state_lock:
            video_state["fps"] = FPS_STATE["fps"]

    with state_lock:
        latest["frame"] = frame
        latest["updated_at"] = datetime.now(timezone.utc).isoformat()

    now = time.monotonic()
    if (now - AI_LAST_RUN["ts"]) < AI_DETECTION_INTERVAL:
        return True

    AI_LAST_RUN["ts"] = now
    _motion_score(frame)

    detections = detector.detect(frame)
    tracks = tracker.update(frame, detections)
    events = behaviour.evaluate(tracks, frame)
    if events:
        alert_manager.process_alerts(events)

    with state_lock:
        latest.update(
            {
                "frame": frame,
                "detections": tracks,
                "events": events,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return True


# ============================================================
# MONITORING LOOP
# ============================================================

def monitoring_loop() -> None:
    """Run AI independently over the newest frame from the capture thread."""
    print("[AI] Processing thread started")

    while not worker_stop.is_set():
        try:
            if camera.is_file_source and camera.capture_finished:

                # Give the dashboard a moment to display
                # the final processed frame.
                time.sleep(0.3)

                with state_lock:
                    video_state.update({
                        "running": False,
                        "status": "CCTV VIDEO FINISHED",
                        "video_finished": True,
                    })

                print("[AI] CCTV video processing finished")
                break

            if not process_frame():
                with state_lock:
                    if video_state["running"]:
                        video_state["status"] = "WAITING FOR CAMERA FRAME"

                time.sleep(0.02)
                continue

            time.sleep(0.02)

        except Exception as exc:
            app.logger.exception(
                "Frame processing failed: %s",
                exc
            )

            with state_lock:
                video_state.update({
                    "status": "AI ERROR"
                })
                latest["detections"] = []
                latest["events"] = []

            time.sleep(0.25)
            if not process_frame():
                with state_lock:
                    if video_state["running"]:
                        video_state["status"] = "WAITING FOR CAMERA FRAME"
                time.sleep(0.02)
                continue
            time.sleep(0.02)
        except Exception as exc:
            app.logger.exception("Frame processing failed: %s", exc)
            with state_lock:
                video_state.update({"status": "AI ERROR"})
            with state_lock:
                latest["detections"] = []
                latest["events"] = []
            time.sleep(0.25)


def start_worker() -> None:
    global worker_thread
    with worker_lock:
        if worker_thread and worker_thread.is_alive():
            return
        worker_stop.clear()
        worker_thread = threading.Thread(target=monitoring_loop, daemon=True)
        worker_thread.start()


def stop_worker() -> None:
    global worker_thread
    worker_stop.set()
    with camera_io_lock:
        camera.release("explicit monitoring stop")
    if worker_thread and worker_thread is not threading.current_thread():
        worker_thread.join(timeout=2)
    worker_thread = None
    with state_lock:
        video_state["running"] = False


# ============================================================
# UPLOAD VALIDATION
# ============================================================

def valid_upload(
    file_storage,
    extensions: set[str],
) -> bool:

    return bool(
        file_storage
        and file_storage.filename
        and "."
        in file_storage.filename
        and file_storage.filename.rsplit(
            ".",
            1,
        )[1].lower()
        in extensions
    )


def unique_upload_path(filename: str) -> Path:
    safe_name = secure_filename(filename)
    stem = Path(safe_name).stem or "cctv_video"
    suffix = Path(safe_name).suffix.lower()
    candidate = UPLOAD_DIR / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def reset_source_state(source: str, filename: str | None = None) -> None:
    with state_lock:
        video_state.update(
            {
                "running": True,
                "status": "RUNNING",
                "fps": 0.0,
                "source": source,
                "video_filename": filename,
                "video_finished": False,
            }
        )
        latest.update(
            {
                "frame": None,
                "detections": [],
                "events": [],
                "updated_at": None,
            }
        )


# ============================================================
# SAVE ANNOTATED EVIDENCE
# ============================================================

def save_annotated_evidence(
    event: dict,
    frame,
    tracks: list[dict],
) -> None:

    if not event.get("evidence"):
        return

    path = (
        EVIDENCE_DIR
        
        / event["evidence"]
    )

    try:

        path.write_bytes(
            frame_to_jpeg(
                frame,
                tracks,
            )
        )

    except (
        OSError,
        TypeError,
    ):

        app.logger.warning(
            "Could not write annotated evidence frame: %s",
            path,
        )


# ============================================================
# SAVE PROCESSED IMAGE
# ============================================================

def save_processed_image(
    frame,
    tracks: list[dict],
) -> str | None:

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    filename = (
        f"image_analysis_{stamp}.jpg"
    )

    try:

        (
            EVIDENCE_DIR / filename
        ).write_bytes(
            frame_to_jpeg(
                frame,
                tracks,
            )
        )

        return filename

    except (
        OSError,
        TypeError,
    ):

        app.logger.warning(
            "Could not write processed CCTV image"
        )

        return None


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/")
def dashboard():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYTICS
# ============================================================

def compute_analytics(
    history: list[dict],
    tracks: list[dict],
    current_events: list[dict],
) -> dict:

    recent_events = history[:10]

    zone_risk = "SAFE"

    if any(
        event.get(
            "event_type",
            "",
        ).lower()
        in {
            "virtual fence violation",
            "restricted zone violation",
            "suspicious movement",
        }
        for event in recent_events
    ):
        zone_risk = "WATCH"

    if current_events:
        zone_risk = "ALERT"

    high_priority = sum(
        1
        for event in history
        if event.get("severity") == "HIGH"
    )

    medium_priority = sum(
        1
        for event in history
        if event.get("severity") == "MEDIUM"
    )

    persons = sum(
        1
        for item in tracks
        if is_person(item)
    )

    vehicles = sum(
        1
        for item in tracks
        if is_vehicle(item)
    )

    return {
        "total_events": len(history),

        "high_priority": high_priority,

        "medium_priority": medium_priority,

        "people": persons,

        "vehicles": vehicles,

        "zone_risk": zone_risk,

        "recent_event_types": [
            event.get("event_type")
            for event in recent_events
            if event.get("event_type")
        ][:5],

        "alert_trend": (
            "UP"
            if current_events or history
            else "STABLE"
        ),
    }
# ============================================================
# VIRTUAL RESTRICTED ZONE API
# ============================================================

@app.get("/api/zone")
def api_get_zone():

    return jsonify({
        "configured": bool(behaviour.zone_points),
        "points": (
            [[x, y] for x, y in behaviour.zone_points]
            if behaviour.zone_points
            else []
        ),
    })


@app.post("/api/zone")
def api_save_zone():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "error": "No zone data received.",
        }), 400

    points = data.get("points")

    if not isinstance(points, list):
        return jsonify({
            "success": False,
            "error": "Zone points must be a list.",
        }), 400

    if len(points) < 3:
        return jsonify({
            "success": False,
            "error": "A restricted zone needs at least 3 points.",
        }), 400

    cleaned_points = []

    try:
        for point in points:

            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
            ):
                raise ValueError("Invalid point format.")

            x = float(point[0])
            y = float(point[1])

            cleaned_points.append([x, y])

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Invalid zone coordinates.",
        }), 400

    # Update BehaviourEngine immediately
    behaviour.zone_points = [
        (point[0], point[1])
        for point in cleaned_points
    ]

    return jsonify({
        "success": True,
        "configured": True,
        "points": cleaned_points,
        "message": "Restricted zone saved successfully.",
    })


@app.delete("/api/zone")
def api_delete_zone():

    behaviour.zone_points = None

    return jsonify({
        "success": True,
        "configured": False,
        "points": [],
        "message": "Restricted zone cleared.",
    })    # --------------------------------------------------------
    # UPDATE BEHAVIOUR ENGINE
    # --------------------------------------------------------

    behaviour.zone_points = [
        (point[0], point[1])
        for point in cleaned_points
    ]

    # Reset previous zone states because
    # the restricted area has changed.
    behaviour.previous_zone_state.clear()
    behaviour.previous_positions.clear()

    # --------------------------------------------------------
    # SAVE TO ENVIRONMENT FILE
    # --------------------------------------------------------

    zone_json = json.dumps(
        cleaned_points
    )

    env_path = BASE_DIR / ".env"

    try:

        existing_lines = []

        if env_path.exists():

            existing_lines = (
                env_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            )

        updated_lines = []

        zone_saved = False

        for line in existing_lines:

            if line.startswith(
                "IBVAP_ZONE_POLYGON="
            ):

                updated_lines.append(
                    f"IBVAP_ZONE_POLYGON={zone_json}"
                )

                zone_saved = True

            else:

                updated_lines.append(line)

        if not zone_saved:

            updated_lines.append(
                f"IBVAP_ZONE_POLYGON={zone_json}"
            )

        env_path.write_text(
            "\n".join(
                updated_lines
            )
            + "\n",
            encoding="utf-8",
        )

    except OSError as exc:

        app.logger.warning(
            "Could not save zone configuration: %s",
            exc,
        )

    return jsonify(
        {
            "success": True,
            "configured": True,
            "points": cleaned_points,
            "message": (
                "Virtual restricted zone saved successfully."
            ),
        }
    )


@app.delete("/api/zone")
def clear_zone():

    behaviour.zone_points = None

    behaviour.previous_zone_state.clear()
    behaviour.previous_positions.clear()

    env_path = BASE_DIR / ".env"

    try:

        if env_path.exists():

            lines = (
                env_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            )

            lines = [
                line
                for line in lines
                if not line.startswith(
                    "IBVAP_ZONE_POLYGON="
                )
            ]

            env_path.write_text(
                "\n".join(lines)
                + (
                    "\n"
                    if lines
                    else ""
                ),
                encoding="utf-8",
            )

    except OSError as exc:

        app.logger.warning(
            "Could not clear zone configuration: %s",
            exc,
        )

    return jsonify(
        {
            "success": True,
            "configured": False,
            "message": (
                "Virtual restricted zone cleared."
            ),
        }
    )

# ============================================================
# STATUS API
# ============================================================

@app.get("/api/status")
def status():

    with state_lock:

        tracks = list(
            latest["detections"]
        )

        current_events = list(
            latest["events"]
        )

        updated_at = latest[
            "updated_at"
        ]

        current_video_state = {
            "running": video_state["running"],
            "status": video_state["status"],
            "fps": video_state["fps"],
            "source": video_state["source"],
            "video_filename": video_state["video_filename"],
            "video_finished": video_state["video_finished"],
        }

    history = database.events()

    people = sum(
        1
        for item in tracks
        if is_person(item)
    )

    vehicles = sum(
        1
        for item in tracks
        if is_vehicle(item)
    )

    analytics = compute_analytics(
        history=history,
        tracks=tracks,
        current_events=current_events,
    )

    alert_list = alert_manager.get_recent(10)

    return jsonify(
        {
            "mode": detector.mode,

            "source": current_video_state["source"],

            "camera_id": "CCTV-01",

            "camera_status": (
                "PROCESSING"
                if current_video_state["running"]
                else current_video_state["status"]
            ),

            "monitoring": current_video_state["running"],

            "video_filename": current_video_state["video_filename"],

            "video_finished": current_video_state["video_finished"],

            "video": {
                **current_video_state,
                "frame_skip": frame_skip,
            },
            "virtual_fence": "NOT CONFIGURED" if not behaviour.zone_points else "CONFIGURED",

            "updated_at": updated_at,

            "counts": {
                "detections": len(tracks),
                "people": people,
                "vehicles": vehicles,
            },

            "active_alerts": len(
                current_events
            ),

            "alerts": alert_list,

            "tracks": tracks,

            "current_events": current_events,

            "history": history,

            "analytics": analytics,
        }
    )


# ============================================================
# CAMERA API
# ============================================================

@app.get("/api/cameras")
def cameras():

    return jsonify(
        [
            {
                "camera_id": "CCTV-01",

                "source": camera.source_label,

                "status": (
                    "PROCESSING"
                    if video_state["running"]
                    else video_state["status"]
                ),
            }
        ]
    )


# ============================================================
# EVENTS
# ============================================================

@app.get("/api/events")
def events():

    return jsonify(database.events())


@app.get("/api/alerts")
def alerts():
    return jsonify(
        alert_manager.get_recent(20)
    )


@app.get("/api/detections")
def detections():
    with state_lock:
        return jsonify(
            list(latest["detections"])
        )


@app.get("/api/analytics")
def analytics():
    with state_lock:
        tracks = list(latest["detections"])
        current_events = list(latest["events"])

    history = database.events()
    return jsonify(
        compute_analytics(
            history=history,
            tracks=tracks,
            current_events=current_events,
        )
    )


# ============================================================
# EVIDENCE
# ============================================================

@app.get("/api/evidence")
def evidence():

    return jsonify(
        [
            {
                "filename": event["evidence"],

                "url": url_for(
                    "evidence_file",
                    filename=event["evidence"],
                ),

                **event,
            }

            for event in database.events()

            if event.get("evidence")
        ]
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/api/image/analyze")
def analyze_image():

    return jsonify({"error": "Image analysis is disabled; use a CCTV source."}), 410

    upload = request.files.get(
        "image"
    )

    if not valid_upload(
        upload,
        ALLOWED_IMAGES,
    ):

        return jsonify(
            {
                "error":
                    "Choose a JPG, JPEG, or PNG CCTV image."
            }
        ), 400

    filename = secure_filename(
        upload.filename
    )

    input_path = (
        UPLOAD_DIR / filename
    )

    try:

        # ----------------------------------------------------
        # SAVE UPLOAD
        # ----------------------------------------------------

        upload.save(
            input_path
        )

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        frame = camera.load_image(
            input_path
        )

        if frame is None:

            return jsonify(
                {
                    "error":
                        "Could not load the uploaded image."
                }
            ), 400

        # ----------------------------------------------------
        # RESET IMAGE TRACKER
        #
        # Image is a separate analysis session.
        # We do not want IDs from an old video to leak
        # into the new uploaded image.
        # ----------------------------------------------------

        tracker.reset()

        # ----------------------------------------------------
        # REAL YOLO DETECTION
        # ----------------------------------------------------

        detections = detector.detect(
            frame
        )

        # ----------------------------------------------------
        # TRACK IMAGE OBJECTS
        # ----------------------------------------------------

        tracks = tracker.update(
            frame,
            detections,
        )

        # ----------------------------------------------------
        # BEHAVIOUR ANALYSIS
        # ----------------------------------------------------

        image_events = behaviour.evaluate(
            tracks,
            frame,
        )

        # ----------------------------------------------------
        # CREATE GENERAL DETECTION EVENT
        # ----------------------------------------------------

        if not image_events and tracks:

            detected_people = [
                item
                for item in tracks
                if is_person(item)
            ]

            detected_vehicles = [
                item
                for item in tracks
                if is_vehicle(item)
            ]

            if detected_people:

                image_events = [
                    behaviour.create_event(
                        "Person detected",
                        detected_people,
                        frame,
                        max(
                            item["confidence"]
                            for item
                            in detected_people
                        ),
                    )
                ]

            elif detected_vehicles:

                image_events = [
                    behaviour.create_event(
                        "Vehicle detected",
                        detected_vehicles,
                        frame,
                        max(
                            item["confidence"]
                            for item
                            in detected_vehicles
                        ),
                    )
                ]

        # ----------------------------------------------------
        # SAVE EVENTS
        # ----------------------------------------------------

        for event in image_events:

            save_annotated_evidence(
                event,
                frame,
                tracks,
            )

        if image_events:
            alert_manager.process_alerts(image_events)

        # ----------------------------------------------------
        # ALWAYS SAVE PROCESSED IMAGE
        # ----------------------------------------------------

        processed_filename = (
            save_processed_image(
                frame,
                tracks,
            )
        )

        # ----------------------------------------------------
        # UPDATE DASHBOARD
        # ----------------------------------------------------

        with state_lock:

            latest.update(
                {
                    "frame": frame,

                    "detections": tracks,

                    "events": image_events,

                    "updated_at": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                }
            )

        # ----------------------------------------------------
        # IMAGE RESULT
        # ----------------------------------------------------

        result_event = next(
            (
                event
                for event in image_events
                if event.get("evidence")
            ),
            None,
        )

        result_filename = (
            result_event["evidence"]
            if result_event
            else processed_filename
        )

        image_url = (
            url_for(
                "evidence_file",
                filename=result_filename,
            )
            if result_filename
            else None
        )

        # ----------------------------------------------------
        # RETURN COMPLETE ANALYSIS
        # ----------------------------------------------------

        return jsonify(
            {
                "mode": detector.mode,

                "source": "uploaded_image",

                "detections": tracks,

                "total_detections": len(
                    tracks
                ),

                "people": sum(
                    1
                    for item in tracks
                    if is_person(item)
                ),

                "vehicles": sum(
                    1
                    for item in tracks
                    if is_vehicle(item)
                ),

                "events": image_events,

                "image_url": image_url,

                "message":
                    "CCTV image analysed successfully.",
            }
        )

    except (
        OSError,
        ValueError,
        TypeError,
    ) as exc:

        app.logger.exception(
            "Image processing failed"
        )

        return jsonify(
            {
                "error":
                    f"Image could not be processed: {exc}"
            }
        ), 400

# ============================================================
# VIRTUAL RESTRICTED ZONE API
# ============================================================


@app.post("/api/zone")
def save_zone():
    data = request.get_json(silent=True) or {}
    points = data.get("points")

    if not isinstance(points, list) or len(points) < 3:
        return jsonify({
            "error": "A restricted zone needs at least 3 points."
        }), 400

    try:
        zone_points = []

        for point in points:
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
            ):
                return jsonify({
                    "error": "Invalid restricted-zone point."
                }), 400

            zone_points.append((
                float(point[0]),
                float(point[1]),
            ))

        behaviour.zone_points = zone_points

        print(
            "[ZONE] Restricted zone saved:",
            behaviour.zone_points
        )

        return jsonify({
            "success": True,
            "configured": True,
            "points": behaviour.zone_points,
            "message": "Restricted zone saved successfully.",
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Zone coordinates must be numbers."
        }), 400


@app.delete("/api/zone")
def delete_zone():
    behaviour.zone_points = None

    print("[ZONE] Restricted zone cleared")

    return jsonify({
        "success": True,
        "configured": False,
        "points": [],
        "message": "Restricted zone cleared.",
    })

# ============================================================
# START VIDEO
# ============================================================

@app.post("/api/video/upload")
def upload_video():
    upload = request.files.get("video")
    if not valid_upload(upload, ALLOWED_VIDEOS):
        return jsonify({
            "success": False,
            "error": "Choose an MP4, AVI, MOV, MKV, or WEBM CCTV video.",
        }), 400

    path = unique_upload_path(upload.filename)
    try:
        upload.save(path)
    except OSError as exc:
        return jsonify({"success": False, "error": f"Video could not be saved: {exc}"}), 400

    return jsonify({
        "success": True,
        "filename": path.name,
        "path": path.name,
        "message": "CCTV video uploaded successfully.",
    })


@app.post("/api/monitoring/start-video")
def start_uploaded_video():
    filename = Path(request.form.get("filename", "")).name
    video_path = UPLOAD_DIR / filename
    if not filename or not video_path.exists() or video_path.suffix.lower().lstrip(".") not in ALLOWED_VIDEOS:
        return jsonify({"error": "Select a valid uploaded CCTV video first."}), 400

    stop_worker()
    if not camera.set_source(str(video_path)):
        return jsonify({"error": "OpenCV could not open the uploaded CCTV video."}), 400

    tracker.reset()
    reset_source_state("CCTV VIDEO", video_path.name)
    start_worker()
    return jsonify({
        "message": "Uploaded CCTV video analysis started.",
        "source": "CCTV VIDEO",
        "video_filename": video_path.name,
        "detection_mode": detector.mode,
        "tracking": "MULTI-OBJECT",
    })

@app.post("/api/video/start")
def start_video():
    upload = request.files.get(
        "video"
    )

    # --------------------------------------------------------
    # UPLOADED VIDEO
    # --------------------------------------------------------

    if upload and upload.filename:

        if not valid_upload(
            upload,
            ALLOWED_VIDEOS,
        ):

            return jsonify(
                {
                    "error":
                        "Choose an MP4, AVI, MOV, or MKV CCTV video."
                }
            ), 400

        filename = secure_filename(
            upload.filename
        )

        video_path = unique_upload_path(filename)

        try:

            upload.save(
                video_path
            )

        except OSError as exc:

            return jsonify(
                {
                    "error":
                        f"Video could not be saved: {exc}"
                }
            ), 400

    # --------------------------------------------------------
    # EXISTING VIDEO
    # --------------------------------------------------------

    else:

        requested = request.form.get("path", "")
        video_path = next(
            (path for path in VIDEO_CANDIDATES if path.exists()),
            VIDEO_FILE,
        )
        if requested:
            candidate = VIDEO_DIR / Path(requested).name
            if candidate.exists():
                video_path = candidate

        if not video_path.exists():

            return jsonify(
                {
                    "error": "No sample CCTV video found. "
                             "Place an MP4 file in video/samples/videos."
                }
            ), 404

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    stop_worker()

    if not camera.set_source(
        str(video_path)
    ):

        return jsonify(
            {
                "error":
                    "OpenCV could not open this CCTV video."
            }
        ), 400

    # --------------------------------------------------------
    # RESET TRACKER
    #
    # New video = new tracking session.
    # --------------------------------------------------------

    tracker.reset()

    # --------------------------------------------------------
    # RESET VIDEO STATE
    # --------------------------------------------------------

    with state_lock:

        video_state.update(
            {
                "running": True,

                "status": "RUNNING",

                "fps": 0.0,

                "source": str(
                    video_path.name
                ),

                "video_filename": str(
                    video_path.name
                ),

                "video_finished": False,
            }
        )

        latest.update(
            {
                "frame": None,
                "detections": [],
                "events": [],
                "updated_at": None,
            }
        )

    start_worker()

    return jsonify(
        {
            "message": (
                "Sample CCTV video monitoring started."
                if not upload
                else "CCTV video monitoring started."
            ),

            "source":
                str(video_path.name),

            "frame_skip":
                frame_skip,

            "detection_mode":
                detector.mode,

            "tracking":
                "MULTI-OBJECT",
        }
    )


# ============================================================
# STOP VIDEO
# ============================================================

@app.post("/api/video/stop")
def stop_video():

    stop_worker()
    camera.set_source(None)
    tracker.reset()

    with state_lock:

        video_state.update(
            {
                "running": False,

                "status": "IDLE",

                "source":
                    camera.source_label,
                "video_filename": None,
                "video_finished": False,
            }
        )

    return jsonify(
        {
            "message":
                "CCTV video monitoring stopped."
        }
    )


@app.post("/api/camera/start")
def start_camera():
    with state_lock:
        if video_state["running"]:
            return jsonify({"message": "Live camera already running.", "source": video_state["source"]})

    stop_worker()
    if not camera.set_source(0):
        return jsonify({
            "error": "Could not access the computer camera. Make sure no other application is using it."
        }), 400
    tracker.reset()
    with state_lock:
        video_state.update({"running": True, "status": "RUNNING", "source": "WEBCAM", "video_filename": None, "video_finished": False})
        latest.update({"frame": None, "detections": [], "events": [], "updated_at": None})
    start_worker()
    return jsonify({"message": "Live camera started.", "source": "WEBCAM"})


@app.post("/api/monitoring/start")
def start_monitoring():
    return start_camera()


@app.post("/api/camera/stop")
def stop_camera():
    return stop_video()


@app.post("/api/monitoring/stop")
def stop_monitoring():
    return stop_video()


@app.post("/api/rtsp/start")
def start_rtsp():
    rtsp_url = request.form.get("url", "").strip()
    if not rtsp_url.lower().startswith(("rtsp://", "rtsps://")):
        return jsonify({"error": "Enter a valid RTSP CCTV URL."}), 400
    stop_worker()
    if not camera.set_source(rtsp_url):
        return jsonify({"error": "RTSP connection failed."}), 400
    tracker.reset()
    with state_lock:
        video_state.update({"running": True, "status": "RUNNING", "source": "RTSP CCTV", "video_filename": None, "video_finished": False})
        latest.update({"frame": None, "detections": [], "events": [], "updated_at": None})
    start_worker()
    return jsonify({"message": "RTSP CCTV started.", "source": "RTSP CCTV"})


@app.post("/api/rtsp/stop")
def stop_rtsp():
    return stop_video()


# ============================================================
# LIVE VIDEO FEED
# ============================================================

@app.get("/video_feed")
def video_feed():

    def stream():
        print("[STREAM] Live stream client connected")

        while True:

            with state_lock:
                if not video_state["running"] and latest["frame"] is None:
                    break

                frame = latest["frame"]
                tracks = list(latest["detections"])

            if frame is None:
                time.sleep(0.03)
                continue

            jpeg = frame_to_jpeg(frame, tracks)

            if not jpeg:
                time.sleep(0.03)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache, no-store, max-age=0\r\n"
                b"Pragma: no-cache\r\n"
                b"Expires: 0\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

            time.sleep(0.03)

        print("[STREAM] Live stream client disconnected")

    return app.response_class(
        stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
# ============================================================
# EVIDENCE FILE
# ============================================================

@app.get("/evidence/<path:filename>")
def evidence_file(
    filename: str,
):

    return send_from_directory(
        EVIDENCE_DIR,
        filename,
    )


# ============================================================
# DEMO SECURITY EVENT
# ============================================================

@app.post("/api/demo-event")
def demo_event():
    return jsonify({"error": "Synthetic events are disabled."}), 410


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host=os.getenv(
            "IBVAP_HOST",
            "0.0.0.0",
        ),
        port=int(
            os.getenv(
                "IBVAP_PORT",
                "5000",
            )
        ),
        debug=False,
        threaded=True,
    ) 