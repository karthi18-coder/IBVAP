from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class EventDatabase:
    """SQLite-backed event and camera storage for the active Flask app."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tracking_id INTEGER,
                    confidence REAL NOT NULL DEFAULT 0,
                    evidence TEXT,
                    evidence_path TEXT,
                    details TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    source TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL
                );
                """
            )

    def add_event(self, event: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO events
                (event_id, event_type, timestamp, camera_id, severity, status,
                 tracking_id, confidence, evidence, evidence_path, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("event_id"),
                    event.get("event_type", "Security event"),
                    event.get("timestamp"),
                    event.get("camera_id", "CCTV-01"),
                    event.get("severity", "MEDIUM"),
                    event.get("status", "NEW"),
                    event.get("tracking_id"),
                    float(event.get("confidence", 0.0)),
                    event.get("evidence"),
                    event.get("evidence_path"),
                    json.dumps(event.get("details", {})),
                ),
            )

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        result = []
        for row in rows:
            event = dict(row)
            event.pop("evidence_path", None)
            try:
                event["details"] = json.loads(event["details"] or "{}")
            except json.JSONDecodeError:
                event["details"] = {}
            result.append(event)
        return result

    def upsert_camera(self, camera_id: str, source: str, status: str, updated_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO cameras(camera_id, source, status, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    source = excluded.source,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (camera_id, source, status, updated_at),
            )
