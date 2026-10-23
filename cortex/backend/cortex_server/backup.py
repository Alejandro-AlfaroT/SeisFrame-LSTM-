"""Backups (SQLite file copies) and exports (JSON, Markdown zip)."""

from __future__ import annotations

import io
import json
import re
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .db import backup_database, connect, schema_version, utcnow

EXPORT_TABLES = ("records", "tags", "relationships", "record_versions", "conflicts", "import_batches", "activity")


def create_backup(conn: sqlite3.Connection, backup_dir: Path, label: str = "manual") -> dict[str, Any]:
    path = backup_database(conn, backup_dir, label)
    return {"path": str(path), "name": path.name, "bytes": path.stat().st_size, "created_at": utcnow()}


def list_backups(backup_dir: Path) -> list[dict[str, Any]]:
    if not backup_dir.exists():
        return []
    return [
        {"name": p.name, "path": str(p), "bytes": p.stat().st_size}
        for p in sorted(backup_dir.glob("cortex-*.db"), reverse=True)
    ]


def export_json(conn: sqlite3.Connection) -> dict[str, Any]:
    """Everything, including trashed records and version history."""
    data: dict[str, Any] = {
        "format": "cortex-export",
        "app_version": __version__,
        "schema_version": schema_version(conn),
        "exported_at": utcnow(),
    }
    for table in EXPORT_TABLES:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        for row in rows:
            for key in ("metadata", "snapshot", "stats"):
                if isinstance(row.get(key), str):
                    row[key] = json.loads(row[key])
        data[table] = rows
    return data


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "-", text.lower()).strip("-")[:60] or "record"


def export_markdown_zip(conn: sqlite3.Connection) -> bytes:
    """One Markdown file per active record with YAML frontmatter, grouped by category."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in conn.execute("SELECT * FROM records WHERE deleted_at IS NULL ORDER BY category, title"):
            tags = [t[0] for t in conn.execute("SELECT tag FROM tags WHERE record_id = ? ORDER BY tag", (r["id"],))]
            links = [
                {"type": e["rel_type"], "target": e["target_id"]}
                for e in conn.execute("SELECT rel_type, target_id FROM relationships WHERE source_id = ?", (r["id"],))
            ]
            front = {
                key: r[key]
                for key in (
                    "id", "title", "category", "kind", "status", "project_id", "source_ref", "confidence",
                    "subject", "predicate", "value", "occurred_at", "created_at", "updated_at",
                )
                if r[key] is not None
            }
            front["verified"] = bool(r["verified"])
            if r["is_demo"]:
                front["demo"] = True
            if tags:
                front["tags"] = tags
            if links:
                front["relationships"] = links
            text = "---\n" + yaml.safe_dump(front, allow_unicode=True, sort_keys=False) + "---\n\n" + r["content"]
            zf.writestr(f"{r['category']}/{_slug(r['title'])}-{r['id'][:8]}.md", text)
        zf.writestr("README.txt", f"CORTEX Markdown export, {utcnow()}. Trashed records are not included.\n")
    return buffer.getvalue()


def restore_backup(backup_file: Path, db_path: Path, backup_dir: Path) -> dict[str, Any]:
    """Replace the live database with a backup. The current database is backed up first.

    Run this only while the server is stopped.
    """
    backup_file = backup_file.resolve(strict=True)
    check = sqlite3.connect(f"file:{backup_file.as_posix()}?mode=ro", uri=True)
    try:
        if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"{backup_file} failed SQLite integrity check")
        if check.execute("SELECT 1 FROM sqlite_master WHERE name = 'records'").fetchone() is None:
            raise ValueError(f"{backup_file} is not a CORTEX database")
    finally:
        check.close()
    safety = None
    if db_path.exists():
        conn = connect(db_path)
        try:
            safety = backup_database(conn, backup_dir, "pre-restore")
        finally:
            conn.close()
        for suffix in ("-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    shutil.copyfile(backup_file, db_path)
    return {"restored_from": str(backup_file), "previous_database_saved_to": str(safety) if safety else None}
