from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from database import EventDatabase


class AlertManager:
    """Structured alert storage for border-surveillance events."""

    def __init__(self, database: EventDatabase | None = None):
        self.alerts: list[dict[str, Any]] = []
        self.database = database

    def process_alerts(self, alerts):
        if not alerts:
            return self.alerts

        for alert in alerts:
            normalized = self._normalize_alert(alert)
            if normalized is not None:
                self.alerts.append(normalized)
                if self.database is not None:
                    self.database.add_event(normalized)
                print("[ALERT]", normalized)
        return self.alerts

    def _normalize_alert(self, alert: Any) -> dict[str, Any] | None:
        if not isinstance(alert, dict):
            return None

        normalized = {
            "event_id": alert.get("event_id"),
            "event_type": alert.get("event_type", "Unknown alert"),
            "severity": str(alert.get("severity", "MEDIUM")).upper(),
            "camera_id": alert.get("camera_id", "CCTV-01"),
            "timestamp": alert.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "tracking_id": alert.get("tracking_id"),
            "confidence": float(alert.get("confidence", 0.0)),
            "status": alert.get("status", "NEW"),
            "evidence": alert.get("evidence"),
            "evidence_path": alert.get("evidence_path"),
            "details": alert.get("details", {}),
        }
        return normalized

    def get_alerts(self):
        return self.alerts

    def clear_alerts(self):
        self.alerts.clear()

    def get_high_priority(self):
        return [alert for alert in self.alerts if alert.get("severity") == "HIGH"]

    def get_recent(self, limit: int = 10):
        if self.database is not None:
            return self.database.events(limit)
        return self.alerts[-limit:]
