"""
IncidentMemory — Shared memory store for all HSE agents.

Responsibilities:
  - Track per-person incident history in real-time (in-memory)
  - Read historical CSV logs for pattern analysis
  - Detect repeat incidents, frequency spikes, and high-risk windows
  - Provide intelligence summaries consumed by agents for decision making
"""

import csv
import os
import time
import datetime
from collections import defaultdict


class IncidentMemory:
    """
    Shared memory layer used by all agents.
    Acts as the 'brain state' of the multi-agent system.
    """

    def __init__(self, csv_log_path: str, running_log_path: str = ""):
        self.csv_log_path = csv_log_path
        self._running_log_path = running_log_path

        # Real-time session memory: {track_id: [(alert_type, timestamp), ...]}
        self._session_log: dict[int, list] = defaultdict(list)

        # Escalation state: {track_id: bool} — tracks if already escalated
        self._escalated: dict[str, bool] = {}

    # ──────────────────────────────────────────
    #  WRITE: Record new incident into session memory
    # ──────────────────────────────────────────

    def record(self, alert_type: str, track_id: int):
        """Log a new incident event into real-time session memory."""
        self._session_log[track_id].append((alert_type, time.time()))

    # ──────────────────────────────────────────
    #  READ: Real-time session analysis
    # ──────────────────────────────────────────

    def get_incident_count(self, track_id: int, alert_type: str, within_seconds: int) -> int:
        """Count how many times a person triggered an alert within a time window."""
        now = time.time()
        cutoff = now - within_seconds
        return sum(
            1 for (atype, ts) in self._session_log[track_id]
            if atype == alert_type and ts >= cutoff
        )

    def is_repeat_offender(self, track_id: int, alert_type: str,
                            threshold: int = 3, within_seconds: int = 3600) -> bool:
        """Returns True if person has triggered the same alert >= threshold times in the window."""
        return self.get_incident_count(track_id, alert_type, within_seconds) >= threshold

    def mark_escalated(self, key: str):
        self._escalated[key] = True

    def is_escalated(self, key: str) -> bool:
        return self._escalated.get(key, False)

    # ──────────────────────────────────────────
    #  READ: Historical CSV analysis
    # ──────────────────────────────────────────

    def get_today_summary(self) -> dict:
        """
        Reads both CSV logs and computes today's incident statistics.
        Used by agents to generate intelligent email summaries.
        """
        today = datetime.date.today().isoformat()
        total = 0
        by_hour = defaultdict(int)
        by_person = defaultdict(int)
        peak_hour = None

        for path in [self.csv_log_path, self._running_log_path]:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or not row[0].startswith(today):
                            continue
                        total += 1
                        try:
                            hour = int(row[0][11:13])
                            by_hour[hour] += 1
                            by_person[row[1]] += 1
                        except (IndexError, ValueError):
                            continue
            except IOError:
                pass

        if by_hour:
            peak_hour = max(by_hour, key=by_hour.get)

        return {
            "total": total,
            "by_hour": dict(by_hour),
            "by_person": dict(by_person),
            "peak_hour": f"{peak_hour:02d}:00" if peak_hour is not None else "N/A",
        }

    def get_historical_summary(self, days: int = 7) -> dict:
        """Reads last N days from both CSVs for weekly trend analysis."""
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        total = 0
        by_day = defaultdict(int)

        for path in [self.csv_log_path, self._running_log_path]:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        try:
                            date_str = row[0][:10]
                            row_date = datetime.date.fromisoformat(date_str)
                            if row_date >= cutoff:
                                total += 1
                                by_day[date_str] += 1
                        except (IndexError, ValueError):
                            continue
            except IOError:
                pass

        worst_day = max(by_day, key=by_day.get) if by_day else "N/A"
        return {"total": total, "by_day": dict(by_day), "worst_day": worst_day}
