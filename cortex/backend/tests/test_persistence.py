"""Information must survive application restarts."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cortex_server.app import create_app
from cortex_server.backup import restore_backup
from cortex_server.config import Settings
from tests.conftest import HEADERS

BACKEND = Path(__file__).resolve().parents[1]


def test_data_survives_recreating_the_app(settings) -> None:
    with TestClient(create_app(settings), headers=HEADERS) as first:
        rec = first.post("/api/records", json={"title": "Persistent fact", "category": "facts", "tags": ["keep"]}).json()
        other = first.post("/api/records", json={"title": "Linked", "category": "sessions"}).json()
        first.post("/api/relationships", json={"source_id": rec["id"], "target_id": other["id"], "rel_type": "supports"})
        first.patch(f"/api/records/{rec['id']}", json={"content": "edited"})
    with TestClient(create_app(settings), headers=HEADERS) as second:
        again = second.get(f"/api/records/{rec['id']}").json()
        assert again["content"] == "edited" and again["tags"] == ["keep"]
        assert again["relationships"]["outgoing"][0]["other"]["title"] == "Linked"
        assert again["version_count"] == 1
        assert second.get("/api/search", params={"q": "persistent"}).json()["total"] == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(db: Path, port: int) -> subprocess.Popen:
    env = {**os.environ, "CORTEX_DB": str(db)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "cortex_server", "serve", "--port", str(port)],
        cwd=BACKEND, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


@pytest.mark.slow
def test_data_survives_a_real_server_process_restart(tmp_path: Path) -> None:
    db = tmp_path / "cortex.db"
    port = _free_port()
    proc = _start(db, port)
    try:
        base = f"http://127.0.0.1:{port}"
        rec = httpx.post(f"{base}/api/records", headers=HEADERS,
                         json={"title": "Survives restart", "category": "projects", "kind": "project"}).json()
        httpx.post(f"{base}/api/demo", headers=HEADERS)
    finally:
        proc.kill()  # abrupt stop: committed transactions must already be durable
        proc.wait(10)
    proc = _start(db, port)
    try:
        base = f"http://127.0.0.1:{port}"
        assert httpx.get(f"{base}/api/records/{rec['id']}").json()["title"] == "Survives restart"
        stats = httpx.get(f"{base}/api/stats").json()
        assert stats["demo_records"] == 11 and stats["open_conflicts"] == 2
    finally:
        proc.kill()
        proc.wait(10)


def test_restore_replaces_database_and_keeps_safety_copy(tmp_path: Path) -> None:
    settings = Settings(db_path=tmp_path / "c.db", backup_dir=tmp_path / "b", frontend_dist=tmp_path / "x")
    with TestClient(create_app(settings), headers=HEADERS) as c:
        c.post("/api/records", json={"title": "Before backup", "category": "facts"})
        backup_path = Path(c.post("/api/backup").json()["path"])
        c.post("/api/records", json={"title": "After backup", "category": "facts"})
    result = restore_backup(backup_path, settings.db_path, settings.backup_dir)
    assert Path(result["previous_database_saved_to"]).exists()
    with TestClient(create_app(settings), headers=HEADERS) as c:
        titles = [r["title"] for r in c.get("/api/search").json()["results"]]
        assert titles == ["Before backup"]
