"""Command-line interface for the HPCA daemon control plane."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hpca.daemon.config import DEFAULT_INBOX, DaemonConfig
from hpca.daemon.control import control_path, get_desired_state
from hpca.daemon.inbox import Inbox
from hpca.daemon.service import (DaemonService, legacy_projects, register_project,
                                 start_project, stop_project, update_project)
from hpca.daemon.slurm import write_wrapper
from hpca.core.config import account_fallback as _account_fallback


def _config(args: argparse.Namespace) -> DaemonConfig:
    script = Path(args.successor_script).resolve() if getattr(args, "successor_script", None) else None
    from hpca.daemon.config import default_allowed_roots
    roots = (tuple(Path(p) for p in getattr(args, "allowed_root", []) or [])
             or default_allowed_roots())
    return DaemonConfig(inbox=Path(args.inbox), allowed_roots=roots,
                        poll_seconds=getattr(args, "poll_seconds", 60), successor_script=script)


def main() -> None:
    parser = argparse.ArgumentParser(prog="hpca-daemon")
    parser.add_argument("--inbox", default=str(DEFAULT_INBOX))
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--account", default=_account_fallback())
    init.add_argument("--wrapper", default=None)

    link = sub.add_parser("link")
    link.add_argument("project_yaml")
    link.add_argument("--project-id")
    link.add_argument("--allowed-root", action="append")

    project_start = sub.add_parser("project-start")
    project_start.add_argument("project_yaml")
    project_start.add_argument("--project-id")
    project_start.add_argument("--allowed-root", action="append")

    project_stop = sub.add_parser("project-stop")
    project_stop.add_argument("project_root")

    project_update = sub.add_parser("project-update")
    project_update.add_argument("project_yaml")
    project_update.add_argument("--project-id")
    project_update.add_argument("--allowed-root", action="append")

    project_status = sub.add_parser("project-status")
    project_status.add_argument("project_root")

    run = sub.add_parser("run")
    run.add_argument("--poll-seconds", type=int, default=60)
    run.add_argument("--allowed-root", action="append")
    run.add_argument("--successor-script")
    run.add_argument("--once", action="store_true")

    migrate = sub.add_parser("migrate-legacy")
    migrate.add_argument("--source", required=True)
    migrate.add_argument("--apply", action="store_true",
                         help="Register resolved projects; default is a read-only preview")
    migrate.add_argument("--allowed-root", action="append")

    sub.add_parser("status")
    args = parser.parse_args()
    config = _config(args)
    if args.command == "init":
        Inbox(config.inbox).initialize()
        wrapper = Path(args.wrapper) if args.wrapper else config.inbox / "hpca-daemon.sbatch"
        print(write_wrapper(wrapper, inbox=config.inbox, account=args.account))
    elif args.command == "link":
        print(register_project(config, Path(args.project_yaml), args.project_id))
    elif args.command == "project-start":
        print(start_project(config, Path(args.project_yaml), args.project_id))
    elif args.command == "project-stop":
        print(stop_project(Path(args.project_root)))
    elif args.command == "project-update":
        print(update_project(config, Path(args.project_yaml), args.project_id))
    elif args.command == "project-status":
        root = Path(args.project_root).resolve(strict=True)
        print(json.dumps({"project_root": str(root),
                          "desired_state": get_desired_state(root),
                          "control": str(control_path(root))}, indent=2))
    elif args.command == "run":
        logging.basicConfig(level=logging.INFO)
        DaemonService(config).run(once=args.once)
    elif args.command == "status":
        path = config.inbox / "daemon.json"
        print(path.read_text() if path.exists() else json.dumps({"state": "NOT_INITIALIZED"}))
    elif args.command == "migrate-legacy":
        for project_id, project_yaml in legacy_projects(Path(args.source)):
            if args.apply:
                print(register_project(config, project_yaml, project_id))
            else:
                print(f"{project_id}\t{project_yaml}")


if __name__ == "__main__":
    main()
