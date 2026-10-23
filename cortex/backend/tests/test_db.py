import sqlite3
from pathlib import Path

import pytest

from cortex_server.db import MIGRATIONS_DIR, connect, migrate, schema_version, transaction


def test_fresh_database_gets_full_schema(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    applied = migrate(db)
    assert applied == [1]
    conn = connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")}
    assert {"records", "tags", "relationships", "record_versions", "conflicts", "activity", "records_fts"} <= tables
    assert schema_version(conn) == 1
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_migrate_is_idempotent_and_does_not_back_up_when_nothing_changes(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    migrate(db, tmp_path / "backups")
    assert migrate(db, tmp_path / "backups") == []
    assert not (tmp_path / "backups").exists()


def test_new_migration_backs_up_existing_data_first(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_initial.sql").write_text((MIGRATIONS_DIR / "0001_initial.sql").read_text("utf-8"), "utf-8")
    db = tmp_path / "c.db"
    migrate(db, tmp_path / "backups", migrations)
    conn = connect(db)
    conn.execute(
        "INSERT INTO records (id, title, category, created_at, updated_at) VALUES ('r1', 'Keep me', 'facts', 'x', 'x')"
    )
    conn.close()

    (migrations / "0002_add_column.sql").write_text("ALTER TABLE records ADD COLUMN pinned INTEGER DEFAULT 0;", "utf-8")
    assert migrate(db, tmp_path / "backups", migrations) == [2]
    backups = list((tmp_path / "backups").glob("cortex-*-pre-migration-0002.db"))
    assert len(backups) == 1
    saved = sqlite3.connect(backups[0])
    assert saved.execute("SELECT title FROM records").fetchone()[0] == "Keep me"
    saved.close()
    conn = connect(db)
    assert schema_version(conn) == 2
    assert conn.execute("SELECT pinned FROM records").fetchone()[0] == 0


def test_failed_migration_rolls_back_completely(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_ok.sql").write_text("CREATE TABLE a (x INTEGER);", "utf-8")
    (migrations / "0002_broken.sql").write_text("CREATE TABLE b (x INTEGER);\nTHIS IS NOT SQL;", "utf-8")
    db = tmp_path / "c.db"
    with pytest.raises(sqlite3.OperationalError):
        migrate(db, tmp_path / "backups", migrations)
    conn = connect(db)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "a" in names and "b" not in names
    assert schema_version(conn) == 1


def test_transaction_rolls_back_on_error_and_nests_as_savepoint(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    conn = connect(db)
    conn.execute("CREATE TABLE t (x INTEGER)")
    with transaction(conn):
        conn.execute("INSERT INTO t VALUES (1)")
        with pytest.raises(RuntimeError):
            with transaction(conn):
                conn.execute("INSERT INTO t VALUES (2)")
                raise RuntimeError("inner failure")
    assert [r[0] for r in conn.execute("SELECT x FROM t")] == [1]
    with pytest.raises(RuntimeError):
        with transaction(conn):
            conn.execute("INSERT INTO t VALUES (3)")
            raise RuntimeError("outer failure")
    assert [r[0] for r in conn.execute("SELECT x FROM t")] == [1]
