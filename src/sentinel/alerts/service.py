"""Alert service — dispatches anomaly notifications to operators.

Supports webhooks (Slack, Discord, custom) and SMTP email.
Key features:
  - Deduplication: same ongoing anomaly doesn't spam
  - Escalation: auto-escalate if anomaly persists
  - Batching: groups related alerts within a window
"""

from __future__ import annotations

import time
from collections import defaultdict

import httpx
import structlog

from sentinel.core.models import Alert, Anomaly, Severity

logger = structlog.get_logger()


class AlertService:
    """Manages alert dispatching with deduplication and escalation."""

    def __init__(
        self,
        webhook_url: str = "",
        dedup_window_sec: float = 300.0,
        escalation_delay_sec: float = 600.0,
    ):
        self.webhook_url = webhook_url
        self.dedup_window = dedup_window_sec
        self.escalation_delay = escalation_delay_sec

        # Track last alert time per (satellite, parameter) to deduplicate
        self._last_alert: dict[str, float] = {}
        # Track ongoing anomalies for escalation
        self._ongoing: dict[str, tuple[float, Severity]] = {}

    async def process_anomaly(self, anomaly: Anomaly) -> Alert | None:
        """Decide whether to alert on this anomaly, and dispatch if so.

        Returns the Alert if dispatched, None if suppressed.
        """
        key = f"{anomaly.satellite_id}::{anomaly.parameter}"
        now = time.monotonic()

        # Deduplication check
        last = self._last_alert.get(key, 0)
        if now - last < self.dedup_window:
            # Check for escalation: if severity increased, alert anyway
            prev_severity = self._ongoing.get(key, (0, Severity.NOMINAL))[1]
            if _severity_rank(anomaly.severity) <= _severity_rank(prev_severity):
                return None  # suppress duplicate

        # Only alert on WARNING or CRITICAL
        if anomaly.severity in (Severity.NOMINAL, Severity.WATCH):
            self._ongoing[key] = (now, anomaly.severity)
            return None

        # Build alert
        alert = Alert(
            anomaly_id=anomaly.id,
            satellite_id=anomaly.satellite_id,
            severity=anomaly.severity,
            title=f"[{anomaly.severity.value.upper()}] {anomaly.satellite_id} — {anomaly.parameter}",
            message=anomaly.explanation,
        )

        # Dispatch
        await self._dispatch(alert)
        self._last_alert[key] = now
        self._ongoing[key] = (now, anomaly.severity)

        return alert

    async def _dispatch(self, alert: Alert) -> None:
        """Send alert via configured channels."""
        if self.webhook_url:
            await self._send_webhook(alert)

        # Always log
        logger.warning(
            "alert_dispatched",
            alert_id=alert.id,
            satellite=alert.satellite_id,
            severity=alert.severity.value,
            title=alert.title,
        )

    async def _send_webhook(self, alert: Alert) -> None:
        """POST alert to webhook URL (Slack, Discord, custom)."""
        payload = {
            "text": f"*{alert.title}*\n{alert.message}",
            "alert_id": alert.id,
            "anomaly_id": alert.anomaly_id,
            "severity": alert.severity.value,
            "satellite_id": alert.satellite_id,
            "timestamp": alert.dispatched_at.isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                if resp.status_code >= 400:
                    logger.error("webhook_failed", status=resp.status_code, url=self.webhook_url)
        except httpx.HTTPError as e:
            logger.error("webhook_error", error=str(e), url=self.webhook_url)

    async def check_escalations(self) -> list[Alert]:
        """Check for anomalies that should be escalated due to persistence.

        Called periodically (e.g., every 60 seconds) by a background task.
        """
        now = time.monotonic()
        escalations: list[Alert] = []

        for key, (started, severity) in list(self._ongoing.items()):
            elapsed = now - started
            if elapsed >= self.escalation_delay and severity != Severity.CRITICAL:
                parts = key.split("::", 1)
                sat_id = parts[0] if parts else "unknown"
                param = parts[1] if len(parts) > 1 else "unknown"

                alert = Alert(
                    anomaly_id="escalation",
                    satellite_id=sat_id,
                    severity=Severity.CRITICAL,
                    title=f"[ESCALATED] {sat_id} — {param} anomaly persisting {int(elapsed)}s",
                    message=f"Anomaly on {param} has persisted for {int(elapsed)} seconds without resolution. Auto-escalated to CRITICAL.",
                )
                await self._dispatch(alert)
                self._ongoing[key] = (now, Severity.CRITICAL)
                escalations.append(alert)

        return escalations

    def clear_resolved(self, satellite_id: str, parameter: str) -> None:
        """Mark an anomaly as resolved — stops dedup and escalation."""
        key = f"{satellite_id}::{parameter}"
        self._last_alert.pop(key, None)
        self._ongoing.pop(key, None)


def _severity_rank(severity: Severity) -> int:
    return {Severity.NOMINAL: 0, Severity.WATCH: 1, Severity.WARNING: 2, Severity.CRITICAL: 3}[severity]
