"""SQLite connections, transactions, migrations and backups."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def later_than(previous: str) -> str:
    """A timestamp strictly after `previous`, so rapid successive writes stay distinguishable."""
    now = utcnow()
    if now > previous:
        return now
    bumped = datetime.fromisoformat(previous.replace("Z", "+00:00")) + timedelta(microseconds=1)
    return bumped.isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex


def connect(db_path: Path) -> sqlite3.Connection:
    # isolation_level=None: transactions are explicit (see `transaction`), never implicit.
    # check_same_thread=False: the web layer may open a connection in one worker thread and use
    # it in another; each connection still serves a single request at a time.
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """All-or-nothing block. Nested use becomes a savepoint inside the outer transaction."""
    if conn.in_transaction:
        name = f"sp_{uuid.uuid4().hex}"
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield conn
        except BaseException:
            conn.execute(f"ROLLBACK TO {name}")
            conn.execute(f"RELEASE {name}")
            raise
        conn.execute(f"RELEASE {name}")
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def backup_database(conn: sqlite3.Connection, backup_dir: Path, label: str = "manual") -> Path:
    """Consistent online copy through SQLite's backup API (safe while the app is running)."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
    target = backup_dir / f"cortex-{stamp}-{safe_label}.db"
    dest = sqlite3.connect(target)
    try:
        conn.backup(dest)
    finally:
        dest.close()
    return target


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, str, Path]]:
    files = []
    for path in directory.glob("*.sql"):
        version_text, _, name = path.stem.partition("_")
        files.append((int(version_text), name, path))
    versions = [version for version, _, _ in files]
    if len(versions) != len(set(versions)):
        raise RuntimeError(f"Duplicate migration versions in {directory}")
    return sorted(files)


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if row is None:
        return 0
    return conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]


def migrate(
    db_path: Path,
    backup_dir: Path | None = None,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list[int]:
    """Apply pending migrations in order, each in its own transaction.

    A database that already holds data is backed up before its schema changes.
    Returns the versions applied.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    existed = db_path.exists() and db_path.stat().st_size > 0
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        pending = [m for m in migration_files(migrations_dir) if m[0] not in applied]
        if pending and existed and applied:
            backup_database(
                conn, backup_dir or db_path.parent / "backups", f"pre-migration-{pending[0][0]:04d}"
            )
        for version, name, path in pending:
            sql = path.read_text(encoding="utf-8")
            try:
                # executescript cannot take parameters; the bookkeeping row is inserted
                # separately inside the same open transaction.
                conn.executescript(f"BEGIN IMMEDIATE;\n{sql}")
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, utcnow()),
                )
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return [version for version, _, _ in pending]
    finally:
        conn.close()
