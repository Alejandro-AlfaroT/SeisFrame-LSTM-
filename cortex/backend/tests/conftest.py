from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from cortex_server.app import create_app
from cortex_server.config import Settings
from cortex_server.db import connect, migrate

FIXTURES = Path(__file__).parent / "fixtures"
HEADERS = {"X-Cortex-Client": "1"}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "cortex.db", backup_dir=tmp_path / "backups", frontend_dist=tmp_path / "no-dist")


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    migrate(settings.db_path, settings.backup_dir)
    c = connect(settings.db_path)
    yield c
    c.close()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings), headers=HEADERS) as c:
        yield c


def make(conn: sqlite3.Connection, **fields) -> str:  # type: ignore[no-untyped-def]
    from cortex_server.repository import create_record

    fields.setdefault("category", "facts")
    fields.setdefault("title", "Untitled")
    return create_record(conn, fields)
