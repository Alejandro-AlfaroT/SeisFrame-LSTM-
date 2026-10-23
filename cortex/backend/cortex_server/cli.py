"""Command line: serve, import, scan, backup, export, restore, demo data.

    python -m cortex_server serve
    python -m cortex_server import-md "<folder>" --route by_type --project "SeisFrame V6" --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import backup, conflicts, demo, importer, repository
from .config import Settings
from .db import connect, migrate


def _open(settings: Settings):  # type: ignore[no-untyped-def]
    migrate(settings.db_path, settings.backup_dir)
    return connect(settings.db_path)


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .app import create_app

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("Refusing to bind to a non-loopback address: CORTEX holds private notes.", file=sys.stderr)
        return 2
    print(f"CORTEX database: {settings.db_path}")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_import(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.folder)
    exclude = tuple(args.exclude) if args.exclude is not None else importer.DEFAULT_EXCLUDED_DIRS
    files, skipped = importer.collect_directory(root, exclude)
    conn = _open(settings)
    try:
        project_id = None
        if args.project:
            if args.dry_run:
                row = conn.execute(
                    "SELECT id FROM records WHERE kind = 'project' AND deleted_at IS NULL AND lower(title) = lower(?)",
                    (args.project,),
                ).fetchone()
                project_id = row["id"] if row else None
                if project_id is None:
                    print(f"(dry run) project '{args.project}' would be created")
            else:
                project_id = importer.ensure_project(conn, args.project)
        options = importer.ImportOptions(
            route=args.route,
            project_id=project_id,
            source_label=args.label or root.resolve().name,
            dry_run=args.dry_run,
            update_edited=args.update_edited,
            exclude_dirs=exclude,
            root=str(root.resolve()),
        )
        report = importer.import_markdown(conn, files, options, skipped)
    finally:
        conn.close()
    if args.verbose:
        _print(report)
    else:
        by_category: dict[str, int] = {}
        for entry in report["files"]:
            if entry["action"] in ("created", "updated"):
                by_category[entry["category"]] = by_category.get(entry["category"], 0) + 1
        _print(
            {
                "dry_run": report["dry_run"],
                "route": report["route"],
                "source": report["source"],
                "counts": report["counts"],
                "by_category": by_category,
                "links": report["links"],
                "skipped": report["skipped"],
                "conflicts": report["conflicts"],
                "warnings": [{"path": e["path"], "warnings": e["warnings"]} for e in report["files"] if e["warnings"]],
            }
        )
    return 0


def cmd_scan(_args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    try:
        result = conflicts.detect(conn)
        _print({**result, "new_conflicts": len(result["new_conflicts"])})
    finally:
        conn.close()
    return 0


def cmd_backup(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    try:
        _print(backup.create_backup(conn, settings.backup_dir, args.label))
    finally:
        conn.close()
    return 0


def cmd_export(args: argparse.Namespace, settings: Settings) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    conn = _open(settings)
    try:
        if args.format == "json":
            out.write_text(json.dumps(backup.export_json(conn), indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            out.write_bytes(backup.export_markdown_zip(conn))
    finally:
        conn.close()
    print(f"wrote {out}")
    return 0


def cmd_restore(args: argparse.Namespace, settings: Settings) -> int:
    if not args.yes:
        print(
            "Restoring replaces the live database (it is backed up first). Stop the server, then re-run"
            " with --yes.",
            file=sys.stderr,
        )
        return 1
    _print(backup.restore_backup(Path(args.backup), settings.db_path, settings.backup_dir))
    return 0


def cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    try:
        _print(demo.load_demo(conn) if args.action == "load" else demo.purge_demo(conn))
    finally:
        conn.close()
    return 0


def cmd_stats(_args: argparse.Namespace, settings: Settings) -> int:
    conn = _open(settings)
    try:
        _print(repository.stats(conn))
    finally:
        conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cortex", description="CORTEX knowledge base")
    parser.add_argument("--db", help="database file (default: cortex/data/cortex.db or $CORTEX_DB)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the API (and the built UI) on localhost")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("import-md", help="import a folder of Markdown notes")
    p.add_argument("folder")
    p.add_argument("--route", choices=importer.ROUTES, default="inbox",
                   help="inbox: everything lands in the Inbox for triage; by_type: file by note type/folder")
    p.add_argument("--project", help="attach imported records to this project (created if missing)")
    p.add_argument("--label", help="source label used in source references (default: folder name)")
    p.add_argument("--exclude", nargs="*", help="folder names to skip (default: Templates)")
    p.add_argument("--update-edited", action="store_true", help="also refresh records edited inside CORTEX")
    p.add_argument("--dry-run", action="store_true", help="report what would happen without writing")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("scan-conflicts", help="look for new candidate contradictions")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("backup", help="copy the database into the backups folder")
    p.add_argument("--label", default="manual")
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser("export", help="export everything as JSON or a Markdown zip")
    p.add_argument("--format", choices=("json", "markdown"), default="json")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("restore", help="replace the database with a backup (server must be stopped)")
    p.add_argument("backup")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("demo", help="load or remove the demonstration records")
    p.add_argument("action", choices=("load", "purge"))
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("stats", help="print dashboard numbers")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args, Settings.from_env(args.db))


if __name__ == "__main__":
    sys.exit(main())
