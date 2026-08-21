"""
alerts.py — Log-based alert system. Writes to JSONL file and stdout.

NO email. All alerts are logged to a JSONL file and printed to stdout.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("hpca.monitor.alerts")

def _alert_log_path() -> Path:
    """Return hpc.alerts_log from platform.yaml (package-local fallback)."""
    from hpca.core.config import Config
    p = Config.get().hpc("alerts_log", "")
    return Path(p) if p else Path(__file__).resolve().parents[2] / "orchestrator" / "logs" / "alerts.jsonl"

ALERT_LOG = _alert_log_path()

# RMSE quality thresholds for alerting
E_RMSE_WARN = 10.0   # meV/atom — warn if above this after training
F_RMSE_WARN = 200.0  # meV/Å   — warn if above this after training


@dataclass
class Alert:
    """Structured alert record emitted by handlers and consumed by the dashboard."""

    timestamp: str
    level: str          # DEBUG INFO WARNING ERROR CRITICAL
    project: str
    handler: str
    message: str
    resolved: bool = False


class AlertEngine:
    """
    Checks project states for anomalies and fires log-based alerts.

    Alert conditions:
    - Handler RUNNING > STUCK_HOURS without progress → WARNING
    - Handler FAILED → ERROR
    - All handlers COMPLETE for a project → INFO
    - MLIP RMSE above threshold after training → WARNING
    - Active Learning cycle >= MAX_CYCLES with poor RMSE → ERROR
    """

    STUCK_HOURS = 96        # handler stuck in RUNNING this long → WARNING

    def check_and_fire(self, all_project_states: list) -> list:
        """
        Check each project state dict for alert conditions.

        Args:
            all_project_states: list of dicts, each containing:
                - "project_dir": str  (project root path)
                - "state": dict       (orchestrator_state.json contents)
                - "updated_at": str   (ISO timestamp of last state update)

        Returns list of Alert dicts that were newly fired this call.
        """
        fired: list[Alert] = []
        now = datetime.now()

        for entry in all_project_states:
            project_dir = entry.get("project_dir", "unknown")
            project_name = Path(project_dir).name
            state = entry.get("state", {})
            handlers = state.get("handlers", {})
            updated_at_str = entry.get("updated_at", "")

            # Parse last update time
            try:
                updated_at = datetime.fromisoformat(updated_at_str) if updated_at_str else None
            except ValueError:
                updated_at = None

            # --- Check each handler ---
            all_handler_stages = []
            for handler_name, h_state in handlers.items():
                stage = h_state.get("stage", "PENDING")
                all_handler_stages.append(stage)

                # FAILED handler → ERROR
                if stage == "FAILED":
                    alert = Alert(
                        timestamp=now.isoformat(),
                        level="ERROR",
                        project=project_name,
                        handler=handler_name,
                        message=f"Handler {handler_name} is FAILED in {project_name}. "
                                f"Check logs: {project_dir}/logs/",
                    )
                    if not self._already_fired(alert):
                        self.fire(alert)
                        fired.append(alert)

                # RUNNING too long → WARNING
                elif stage == "RUNNING":
                    started_at_str = h_state.get("started_at", "")
                    try:
                        started_at = datetime.fromisoformat(started_at_str) if started_at_str else None
                    except ValueError:
                        started_at = None

                    if started_at and (now - started_at) > timedelta(hours=self.STUCK_HOURS):
                        hours_running = (now - started_at).total_seconds() / 3600
                        alert = Alert(
                            timestamp=now.isoformat(),
                            level="WARNING",
                            project=project_name,
                            handler=handler_name,
                            message=(
                                f"Handler {handler_name} in {project_name} has been RUNNING "
                                f"for {hours_running:.0f}h (threshold={self.STUCK_HOURS}h). "
                                "Job may be stuck or failed silently."
                            ),
                        )
                        if not self._already_fired(alert):
                            self.fire(alert)
                            fired.append(alert)

            # --- All handlers complete → INFO ---
            non_pending = [s for s in all_handler_stages if s != "PENDING"]
            if non_pending and all(s in ("COMPLETE", "SKIPPED") for s in non_pending):
                alert = Alert(
                    timestamp=now.isoformat(),
                    level="INFO",
                    project=project_name,
                    handler="orchestrator",
                    message=f"Project {project_name} workflow COMPLETE. All handlers finished.",
                )
                if not self._already_fired(alert):
                    self.fire(alert)
                    fired.append(alert)

            # --- MLIP RMSE quality check ---
            mlip_state = handlers.get("h04_mlip", {})
            if mlip_state.get("stage") == "COMPLETE":
                e_rmse = mlip_state.get("e_rmse")
                f_rmse = mlip_state.get("f_rmse")
                if e_rmse is not None and e_rmse > E_RMSE_WARN * 1e-3:  # state stores eV/atom
                    alert = Alert(
                        timestamp=now.isoformat(),
                        level="WARNING",
                        project=project_name,
                        handler="h04_mlip",
                        message=(
                            f"MLIP for {project_name} has high E_RMSE={e_rmse*1000:.2f} meV/atom "
                            f"(threshold={E_RMSE_WARN} meV/atom). Active learning may be needed."
                        ),
                    )
                    if not self._already_fired(alert):
                        self.fire(alert)
                        fired.append(alert)

            # --- Active Learning max cycles with poor RMSE → ERROR ---
            al_state = handlers.get("h13_active_learning", {})
            if (al_state.get("stage") == "COMPLETE"
                    and not al_state.get("converged")
                    and al_state.get("cycle", 0) >= 3):
                e_rmse = al_state.get("e_rmse", float("inf"))
                f_rmse = al_state.get("f_rmse", float("inf"))
                alert = Alert(
                    timestamp=now.isoformat(),
                    level="ERROR",
                    project=project_name,
                    handler="h13_active_learning",
                    message=(
                        f"Active learning for {project_name} exhausted {al_state.get('cycle')} cycles "
                        f"without convergence. Final: E_RMSE={e_rmse:.2f} meV/atom, "
                        f"F_RMSE={f_rmse:.2f} meV/Å. Manual intervention needed."
                    ),
                )
                if not self._already_fired(alert):
                    self.fire(alert)
                    fired.append(alert)

        return [asdict(a) for a in fired]

    def fire(self, alert: Alert) -> None:
        """Write alert to JSONL log file and print to stdout."""
        # Ensure log directory exists
        ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)

        record = asdict(alert)
        line = json.dumps(record)

        # Append to JSONL
        try:
            with ALERT_LOG.open("a") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            log.error("Cannot write to alert log %s: %s", ALERT_LOG, exc)

        # Also log to Python logging
        level_map = {
            "DEBUG":    logging.DEBUG,
            "INFO":     logging.INFO,
            "WARNING":  logging.WARNING,
            "ERROR":    logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        py_level = level_map.get(alert.level, logging.INFO)
        log.log(py_level, "[ALERT][%s][%s] %s", alert.project, alert.handler, alert.message)

        # Print to stdout for immediate visibility
        print(f"[ALERT {alert.level}] [{alert.project}/{alert.handler}] {alert.message}")

    def get_recent(self, hours: int = 24) -> list:
        """Return alerts from the last N hours."""
        if not ALERT_LOG.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        results = []

        try:
            for line in ALERT_LOG.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    ts = datetime.fromisoformat(record.get("timestamp", ""))
                    if ts >= cutoff:
                        results.append(record)
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception as exc:
            log.warning("Cannot read alert log: %s", exc)

        return results

    def get_unresolved(self) -> list:
        """Return all unresolved alerts."""
        if not ALERT_LOG.exists():
            return []

        results = []
        try:
            for line in ALERT_LOG.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not record.get("resolved", False):
                        results.append(record)
                except json.JSONDecodeError:
                    continue
        except Exception as exc:
            log.warning("Cannot read alert log: %s", exc)

        return results

    def resolve(self, project: str, handler: str) -> None:
        """Mark all unresolved alerts for (project, handler) as resolved."""
        if not ALERT_LOG.exists():
            return

        try:
            lines = ALERT_LOG.read_text().splitlines()
            new_lines = []
            resolved_count = 0

            for line in lines:
                if not line.strip():
                    new_lines.append(line)
                    continue
                try:
                    record = json.loads(line)
                    if (record.get("project") == project
                            and record.get("handler") == handler
                            and not record.get("resolved", False)):
                        record["resolved"] = True
                        resolved_count += 1
                    new_lines.append(json.dumps(record))
                except json.JSONDecodeError:
                    new_lines.append(line)

            ALERT_LOG.write_text("\n".join(new_lines) + "\n")
            if resolved_count:
                log.info("Resolved %d alerts for %s/%s", resolved_count, project, handler)
        except Exception as exc:
            log.warning("Cannot update alert log: %s", exc)

    def _already_fired(self, alert: Alert) -> bool:
        """Check if an equivalent unresolved alert already exists (deduplication)."""
        if not ALERT_LOG.exists():
            return False

        # Only deduplicate within the last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        try:
            for line in ALERT_LOG.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("resolved"):
                        continue
                    ts_str = record.get("timestamp", "")
                    ts = datetime.fromisoformat(ts_str) if ts_str else None
                    if ts and ts < cutoff:
                        continue
                    # Same project + handler + level = duplicate
                    if (record.get("project") == alert.project
                            and record.get("handler") == alert.handler
                            and record.get("level") == alert.level):
                        return True
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception:
            pass
        return False
