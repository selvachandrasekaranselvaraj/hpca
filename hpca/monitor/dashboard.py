"""
dashboard.py — HPCA monitoring web dashboard.

Auto-detects available framework: FastAPI (preferred) → Flask → stdlib http.server.
Provides:
  GET  /           → HTML project status dashboard (auto-refreshes every 60s)
  GET  /api/status → JSON all project states
  GET  /api/jobs   → JSON squeue output for $USER
  GET  /api/log    → last 100 lines of orchestrator log
  POST /api/advance → write /tmp/hpca_advance_trigger to nudge orchestrator

Run:
    python -m hpca.monitor.dashboard --port 8050 --root /path/to/workspace
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("hpca.monitor.dashboard")

# ── Framework detection ────────────────────────────────────────────────────────

try:
    import fastapi as _fastapi
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import flask as _flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# ── Constants ──────────────────────────────────────────────────────────────────

ORCH_LOG_GLOB = "orchestrator/logs/hpca_orch_*.log"
ADVANCE_TRIGGER = Path("/tmp/hpca_advance_trigger")

STAGE_COLORS = {
    "COMPLETE": "#27ae60",   # green
    "RUNNING":  "#f39c12",   # orange
    "FAILED":   "#e74c3c",   # red
    "PENDING":  "#95a5a6",   # gray
    "SKIPPED":  "#3498db",   # blue
}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="refresh" content="60">
  <title>HPCA Dashboard — {timestamp}</title>
  <style>
    body {{ font-family: 'Courier New', monospace; background: #1a1a2e; color: #e0e0e0;
           margin: 0; padding: 16px; }}
    h1   {{ color: #00d4ff; margin-bottom: 4px; }}
    h2   {{ color: #a0c4ff; margin-top: 24px; border-bottom: 1px solid #333; padding-bottom: 4px; }}
    .ts  {{ color: #888; font-size: 0.85em; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; font-size: 0.88em; }}
    th   {{ background: #16213e; color: #a0c4ff; padding: 8px 12px; text-align: left;
            border-bottom: 2px solid #00d4ff; }}
    td   {{ padding: 6px 12px; border-bottom: 1px solid #2a2a4a; vertical-align: top; }}
    tr:hover td {{ background: #16213e; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
              font-size: 0.80em; font-weight: bold; color: #fff; }}
    .COMPLETE {{ background: #27ae60; }}
    .RUNNING  {{ background: #f39c12; }}
    .FAILED   {{ background: #e74c3c; }}
    .PENDING  {{ background: #555; }}
    .SKIPPED  {{ background: #3498db; }}
    .alert-ERROR    {{ color: #e74c3c; }}
    .alert-WARNING  {{ color: #f39c12; }}
    .alert-INFO     {{ color: #27ae60; }}
    pre  {{ background: #0d0d1a; padding: 12px; border-radius: 4px;
            overflow-x: auto; font-size: 0.80em; max-height: 300px; overflow-y: scroll; }}
    button {{ background: #00d4ff; color: #000; border: none; padding: 8px 16px;
              border-radius: 4px; cursor: pointer; font-weight: bold; }}
    button:hover {{ background: #00b8d9; }}
    .progress {{ background: #2a2a4a; border-radius: 4px; height: 10px; min-width: 80px; display: inline-block; }}
    .progress-bar {{ background: #27ae60; height: 10px; border-radius: 4px; }}
    a {{ color: #a0c4ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>HPCA Autonomous Platform</h1>
  <p class="ts">Updated: {timestamp} &nbsp;|&nbsp; Auto-refresh: 60s
    &nbsp;|&nbsp; <a href="/api/status">JSON</a>
    &nbsp;|&nbsp; <a href="/api/jobs">Jobs</a>
    &nbsp;|&nbsp; <a href="/api/log">Log</a>
    &nbsp;|&nbsp;
    <form style="display:inline" method="post" action="/api/advance">
      <button type="submit">Advance Orchestrator</button>
    </form>
  </p>

  <h2>Projects</h2>
  {projects_table}

  <h2>Active Slurm Jobs</h2>
  <pre>{jobs_text}</pre>

  <h2>Recent Alerts</h2>
  {alerts_html}

  <h2>Orchestrator Log (last 50 lines)</h2>
  <pre>{log_tail}</pre>
</body>
</html>
"""


# ── Data collection helpers ────────────────────────────────────────────────────

def _discover_projects(root: Path) -> List[Dict[str, Any]]:
    """Find all project.yaml files under root, load state and project info."""
    projects = []
    for py_path in sorted(root.glob("*/project.yaml")):
        proj_dir = py_path.parent
        proj_name = proj_dir.name

        # Load project.yaml
        proj_yaml: dict = {}
        try:
            import yaml
            proj_yaml = yaml.safe_load(py_path.read_text()) or {}
        except Exception:
            proj_yaml = {}

        # Load orchestrator state
        state_file = proj_dir / "logs" / "orchestrator_state.json"
        state: dict = {}
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
            except Exception:
                pass

        # Count handler stages
        handlers = state.get("handlers", {})
        stage_counts = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0, "PENDING": 0, "SKIPPED": 0}
        for h_state in handlers.values():
            s = h_state.get("stage", "PENDING")
            stage_counts[s] = stage_counts.get(s, 0) + 1
        total_handlers = len(handlers)

        # Active jobs
        active_jobs = []
        for h_name, h_state in handlers.items():
            if h_state.get("stage") == "RUNNING":
                job = h_state.get("job")
                if job:
                    active_jobs.append(f"{h_name}:{job}")

        projects.append({
            "name": proj_name,
            "full_name": proj_yaml.get("full_name", proj_name),
            "category": proj_yaml.get("category", "unknown"),
            "D_best": proj_yaml.get("D_best") or proj_yaml.get("D_aimd"),
            "Ea_best": proj_yaml.get("Ea_best") or proj_yaml.get("Ea_aimd"),
            "stage_counts": stage_counts,
            "total_handlers": total_handlers,
            "active_jobs": active_jobs,
            "updated": state.get("updated", ""),
            "handlers": handlers,
            "state": state,
            "project_dir": str(proj_dir),
        })
    return projects


def _get_squeue(user: str | None = None) -> str:
    """Run squeue and return output string."""
    try:
        import os
        user = user or os.environ.get("USER", "")
        cmd = ["squeue", "-u", user,
               "--format=%.10i %.25j %.8T %.12M %.6D %Z",
               "--noheader"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or "(no jobs)"
    except Exception as exc:
        return f"(squeue error: {exc})"


def _get_log_tail(root: Path, n: int = 100) -> str:
    """Return last N lines of most recent orchestrator log."""
    hpca_root = root / "hpca"
    log_files = sorted(hpca_root.glob(ORCH_LOG_GLOB)) if hpca_root.exists() else []
    # Also check relative to root
    log_files += sorted(root.glob(ORCH_LOG_GLOB))
    # Deduplicate
    seen = set()
    unique = []
    for lf in log_files:
        if lf not in seen:
            seen.add(lf)
            unique.append(lf)

    if not unique:
        return "(no orchestrator log found)"
    try:
        latest = unique[-1]
        lines = latest.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(log read error: {exc})"


def _build_projects_table(projects: List[Dict]) -> str:
    """Build HTML table for project list."""
    if not projects:
        return "<p>No projects found.</p>"

    rows = []
    for p in projects:
        sc = p["stage_counts"]
        total = p["total_handlers"]
        complete = sc.get("COMPLETE", 0)
        failed = sc.get("FAILED", 0)
        running = sc.get("RUNNING", 0)
        pct = int(100 * complete / total) if total else 0

        progress_html = (
            f'<div class="progress" title="{complete}/{total} handlers complete">'
            f'<div class="progress-bar" style="width:{pct}%"></div></div>'
            f' {pct}%'
        )

        badges = []
        for stage, count in sc.items():
            if count > 0:
                badges.append(f'<span class="badge {stage}">{count} {stage}</span>')

        active_jobs_str = "<br>".join(p["active_jobs"]) if p["active_jobs"] else "—"

        D_str = f"{p['D_best']:.2e}" if p["D_best"] else "—"
        Ea_str = f"{p['Ea_best']:.3f} eV" if p["Ea_best"] else "—"

        updated = p["updated"][:19].replace("T", " ") if p["updated"] else "—"

        rows.append(
            f"<tr>"
            f"<td><strong>{p['name']}</strong><br><small>{p['full_name']}</small></td>"
            f"<td>{p['category']}</td>"
            f"<td>{progress_html}<br>{'  '.join(badges)}</td>"
            f"<td>{D_str}</td>"
            f"<td>{Ea_str}</td>"
            f"<td><small>{active_jobs_str}</small></td>"
            f"<td><small>{updated}</small></td>"
            f"</tr>"
        )

    header = (
        "<table><thead><tr>"
        "<th>Project</th><th>Category</th><th>Progress</th>"
        "<th>D (m²/s)</th><th>Ea</th><th>Active Jobs</th><th>Updated</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return header


def _build_alerts_html(alerts: list) -> str:
    """Build HTML for recent alerts."""
    if not alerts:
        return "<p style='color:#27ae60'>No recent alerts.</p>"

    rows = []
    for a in reversed(alerts[-20:]):  # show last 20, newest first
        level = a.get("level", "INFO")
        ts = a.get("timestamp", "")[:19].replace("T", " ")
        resolved = " ✓" if a.get("resolved") else ""
        rows.append(
            f'<tr class="alert-{level}">'
            f"<td>{ts}</td>"
            f"<td><strong>{level}</strong></td>"
            f"<td>{a.get('project', '')}</td>"
            f"<td>{a.get('handler', '')}</td>"
            f"<td>{a.get('message', '')}{resolved}</td>"
            f"</tr>"
        )

    return (
        "<table><thead><tr>"
        "<th>Time</th><th>Level</th><th>Project</th><th>Handler</th><th>Message</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _build_html(root: Path) -> str:
    """Build the full dashboard HTML."""
    projects = _discover_projects(root)
    jobs_text = _get_squeue()
    log_tail = _get_log_tail(root)

    from .alerts import AlertEngine
    engine = AlertEngine()
    alerts = engine.get_recent(hours=48)
    alerts_html = _build_alerts_html(alerts)
    projects_table = _build_projects_table(projects)

    return HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        projects_table=projects_table,
        jobs_text=jobs_text,
        alerts_html=alerts_html,
        log_tail=log_tail[-8000:],   # cap HTML size
    )


# ── FastAPI app ────────────────────────────────────────────────────────────────

def create_app(root: Path):
    """Create and return the web application (FastAPI or Flask or stdlib)."""
    root = Path(root)

    if HAS_FASTAPI:
        return _create_fastapi_app(root)
    elif HAS_FLASK:
        return _create_flask_app(root)
    else:
        return _create_stdlib_app(root)


def _create_fastapi_app(root: Path):
    """Create FastAPI app."""
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="HPCA Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the auto-refreshing HTML dashboard page."""
        html = _build_html(root)
        return HTMLResponse(content=html)

    @app.get("/api/status")
    async def api_status():
        """Return JSON summary of all discovered project states."""
        projects = _discover_projects(root)
        # Strip heavy handler data for JSON response
        for p in projects:
            p.pop("state", None)
        return JSONResponse(content={"projects": projects,
                                     "timestamp": datetime.now().isoformat()})

    @app.get("/api/jobs")
    async def api_jobs():
        """Return JSON-wrapped squeue output for the current user."""
        import os
        jobs = _get_squeue(os.environ.get("USER"))
        return JSONResponse(content={"jobs": jobs,
                                     "timestamp": datetime.now().isoformat()})

    @app.get("/api/log")
    async def api_log():
        """Return the last 100 lines of the most recent orchestrator log as JSON."""
        tail = _get_log_tail(root, n=100)
        return JSONResponse(content={"log": tail,
                                     "timestamp": datetime.now().isoformat()})

    @app.post("/api/advance")
    async def api_advance():
        """Write the advance trigger file to nudge the orchestrator forward one cycle."""
        ADVANCE_TRIGGER.write_text(datetime.now().isoformat())
        log.info("Advance trigger written to %s", ADVANCE_TRIGGER)
        return JSONResponse(content={"status": "ok",
                                     "trigger": str(ADVANCE_TRIGGER)})

    return app


def _create_flask_app(root: Path):
    """Create Flask app (fallback)."""
    from flask import Flask, jsonify, request, Response

    app = Flask("hpca_dashboard")

    @app.route("/")
    def index():
        """Serve the auto-refreshing HTML dashboard page."""
        html = _build_html(root)
        return Response(html, mimetype="text/html")

    @app.route("/api/status")
    def api_status():
        """Return JSON summary of all discovered project states."""
        projects = _discover_projects(root)
        for p in projects:
            p.pop("state", None)
        return jsonify({"projects": projects, "timestamp": datetime.now().isoformat()})

    @app.route("/api/jobs")
    def api_jobs():
        """Return JSON-wrapped squeue output for the current user."""
        import os
        return jsonify({"jobs": _get_squeue(os.environ.get("USER")),
                        "timestamp": datetime.now().isoformat()})

    @app.route("/api/log")
    def api_log():
        """Return the last 100 lines of the most recent orchestrator log as JSON."""
        return jsonify({"log": _get_log_tail(root, n=100),
                        "timestamp": datetime.now().isoformat()})

    @app.route("/api/advance", methods=["POST"])
    def api_advance():
        """Write the advance trigger file to nudge the orchestrator forward one cycle."""
        ADVANCE_TRIGGER.write_text(datetime.now().isoformat())
        return jsonify({"status": "ok", "trigger": str(ADVANCE_TRIGGER)})

    return app


def _create_stdlib_app(root: Path):
    """Create minimal stdlib http.server based app (last resort)."""
    import http.server
    import urllib.parse

    class Handler(http.server.BaseHTTPRequestHandler):
        """Minimal stdlib HTTP request handler serving all dashboard routes."""

        def log_message(self, fmt, *args):
            """Route BaseHTTPRequestHandler access logs to the hpca logger."""
            log.debug("HTTP %s", fmt % args)

        def _send_json(self, data: dict, code: int = 200):
            """Serialise data to JSON and write the HTTP response."""
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str):
            """Encode html and write an HTTP 200 text/html response."""
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """Dispatch GET requests to /, /api/status, /api/jobs, and /api/log."""
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", ""):
                self._send_html(_build_html(root))
            elif path == "/api/status":
                projects = _discover_projects(root)
                for p in projects:
                    p.pop("state", None)
                self._send_json({"projects": projects})
            elif path == "/api/jobs":
                import os
                self._send_json({"jobs": _get_squeue(os.environ.get("USER"))})
            elif path == "/api/log":
                self._send_json({"log": _get_log_tail(root, n=100)})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            """Dispatch POST requests; handles /api/advance by writing the trigger file."""
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/advance":
                ADVANCE_TRIGGER.write_text(datetime.now().isoformat())
                self._send_json({"status": "ok"})
            else:
                self._send_json({"error": "not found"}, 404)

    return Handler


# ── Main entry point ───────────────────────────────────────────────────────────

def main():
    """CLI entry point: hpca-monitor."""
    parser = argparse.ArgumentParser(
        description="HPCA monitoring dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",   type=int, default=8050,
                        help="Port to listen on")
    parser.add_argument("--host",   default="0.0.0.0",
                        help="Host/bind address")
    from hpca.core.config import Config as _Cfg
    parser.add_argument("--root",   default=_Cfg.get().hpc("project_base", str(Path.cwd())),
                        help="Root directory containing project subdirectories")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
    )

    root = Path(args.root)
    if not root.exists():
        log.error("Root directory %s does not exist", root)
        sys.exit(1)

    log.info("HPCA Dashboard starting on http://%s:%d", args.host, args.port)
    log.info("Scanning projects under: %s", root)
    log.info("Framework: %s", "FastAPI" if HAS_FASTAPI else ("Flask" if HAS_FLASK else "stdlib"))

    if HAS_FASTAPI:
        import uvicorn
        app = _create_fastapi_app(root)
        uvicorn.run(app, host=args.host, port=args.port,
                    log_level=args.log_level.lower())

    elif HAS_FLASK:
        app = _create_flask_app(root)
        app.run(host=args.host, port=args.port, debug=False)

    else:
        # stdlib fallback
        import http.server
        HandlerClass = _create_stdlib_app(root)
        server = http.server.HTTPServer((args.host, args.port), HandlerClass)
        log.info("Serving on http://%s:%d (stdlib http.server)", args.host, args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log.info("Dashboard stopped")


if __name__ == "__main__":
    main()
