import io
import json
import zipfile

from fastapi.testclient import TestClient

from cortex_server.app import create_app


def test_writes_require_the_client_header(settings) -> None:
    with TestClient(create_app(settings)) as bare:
        r = bare.post("/api/records", json={"title": "x", "category": "facts"})
        assert r.status_code == 403
        assert bare.get("/api/stats").status_code == 200  # reads are fine


def test_foreign_host_header_is_rejected(client) -> None:
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400


def test_record_lifecycle_over_http(client) -> None:
    project = client.post("/api/records", json={"title": "Exhibit", "category": "projects", "kind": "project"}).json()
    created = client.post(
        "/api/records",
        json={
            "title": "Lamp decision",
            "category": "projects",
            "kind": "decision",
            "project_id": project["id"],
            "content": "Use an LED replica.",
            "tags": ["lighting"],
            "links": [{"target_id": project["id"], "rel_type": "part_of"}],
        },
    )
    assert created.status_code == 201
    rec = created.json()
    assert rec["relationships"]["outgoing"][0]["rel_type"] == "part_of"

    patched = client.patch(f"/api/records/{rec['id']}", json={"status": "accepted", "expected_updated_at": rec["updated_at"]})
    assert patched.status_code == 200 and patched.json()["status"] == "accepted"
    stale = client.patch(f"/api/records/{rec['id']}", json={"status": "x", "expected_updated_at": rec["updated_at"]})
    assert stale.status_code == 409

    assert client.get(f"/api/records/{rec['id']}/versions").json()[0]["snapshot"]["status"] is None
    assert client.delete(f"/api/records/{rec['id']}").json()["deleted"] is True
    assert client.delete(f"/api/records/{rec['id']}/purge").status_code == 422  # needs confirm
    assert client.post(f"/api/records/{rec['id']}/restore").json()["deleted"] is False
    client.delete(f"/api/records/{rec['id']}")
    assert [t["id"] for t in client.get("/api/trash").json()] == [rec["id"]]
    assert client.delete(f"/api/records/{rec['id']}/purge", params={"confirm": rec["id"]}).status_code == 200
    assert client.get(f"/api/records/{rec['id']}").status_code == 404


def test_validation_errors_are_422(client) -> None:
    assert client.post("/api/records", json={"title": "x", "category": "nope"}).status_code == 422
    assert client.post("/api/records", json={"title": "x", "category": "facts", "unknown": 1}).status_code == 422
    assert client.post("/api/relationships", json={"source_id": "a", "target_id": "b", "rel_type": "likes"}).status_code in (404, 422)


def test_search_endpoint_with_filters(client) -> None:
    client.post("/api/records", json={"title": "IMK modes", "category": "facts", "tags": ["imk"], "content": "post-capping"})
    client.post("/api/records", json={"title": "Daily", "category": "sessions", "content": "imk study", "occurred_at": "2026-10-22"})
    r = client.get("/api/search", params={"q": "imk", "category": ["facts"]}).json()
    assert [x["title"] for x in r["results"]] == ["IMK modes"]
    r = client.get("/api/search", params={"tag": ["imk"]}).json()
    assert r["total"] == 1
    r = client.get("/api/search", params={"q": "imk study", "mode": "similar"}).json()
    assert r["method"] == "tfidf-cosine" and r["results"][0]["title"] == "Daily"


def test_demo_conflict_review_flow(client) -> None:
    loaded = client.post("/api/demo").json()
    assert loaded["created"] == 11 and loaded["conflicts"] == 2
    assert client.post("/api/demo").json()["created"] == 0  # idempotent
    stats = client.get("/api/stats").json()
    assert stats["demo_records"] == 11 and stats["open_conflicts"] == 2
    conflicts = client.get("/api/conflicts", params={"status": "open"}).json()
    structured = next(c for c in conflicts if c["reason"] == "structured")
    detail = client.get(f"/api/conflicts/{structured['id']}").json()
    assert detail["record_a"]["source_ref"].startswith("DEMO:") and detail["record_b"]["source_ref"].startswith("DEMO:")
    resolved = client.post(f"/api/conflicts/{structured['id']}/resolve", json={"resolution": "prefer_b", "note": "survey"}).json()
    assert resolved["status"] == "resolved"
    assert client.get("/api/stats").json()["open_conflicts"] == 1
    reopened = client.post(f"/api/conflicts/{structured['id']}/reopen").json()
    assert reopened["status"] == "open"

    brain = client.get("/api/brain").json()
    assert [r["region"] for r in brain["regions"]] == ["prefrontal", "frontal", "temporal", "hippocampus", "occipital"]
    frontal = next(r for r in brain["regions"] if r["region"] == "frontal")
    assert frontal["count"] == 2 and frontal["clusters"][0]["conflict_ids"]
    assert len(brain["nodes"]) == 11 and len(brain["edges"]) == 7
    assert client.get("/api/brain", params={"include_demo": False}).json()["nodes"] == []

    removed = client.delete("/api/demo").json()
    assert removed == {"removed": 11}
    assert client.get("/api/stats").json()["total_records"] == 0


def test_graph_neighbourhood(client) -> None:
    ids = [client.post("/api/records", json={"title": t, "category": "facts"}).json()["id"] for t in "ABCD"]
    client.post("/api/relationships", json={"source_id": ids[0], "target_id": ids[1], "rel_type": "supports"})
    client.post("/api/relationships", json={"source_id": ids[1], "target_id": ids[2], "rel_type": "depends_on"})
    client.post("/api/relationships", json={"source_id": ids[2], "target_id": ids[3], "rel_type": "relates_to"})
    g = client.get("/api/graph", params={"record_id": ids[0], "depth": 2}).json()
    assert {n["title"] for n in g["nodes"]} == {"A", "B", "C"}
    assert {e["rel_type"] for e in g["edges"]} == {"supports", "depends_on"}
    rel = g["edges"][0]["id"]
    assert client.patch(f"/api/relationships/{rel}", json={"rel_type": "contradicts"}).status_code == 200
    assert client.delete(f"/api/relationships/{rel}").status_code == 200


def test_markdown_upload_import_preview_then_commit(client) -> None:
    files = [
        {"path": "notes/Idea.md", "content": "---\ntype: concept\n---\n# Idea\n\nSee [[Other]]."},
        {"path": "notes/Other.md", "content": "# Other\n\nPlain note."},
        {"path": "../escape.md", "content": "nope"},
        {"path": "Templates/T.md", "content": "# T"},
    ]
    preview = client.post("/api/import/markdown", json={"files": files, "route": "inbox", "dry_run": True}).json()
    assert preview["counts"] == {"created": 2} and preview["dry_run"] is True
    assert {s["path"] for s in preview["skipped"]} == {"../escape.md", "Templates/T.md"}
    assert client.get("/api/stats").json()["total_records"] == 0
    done = client.post("/api/import/markdown", json={"files": files, "route": "inbox", "dry_run": False, "source_label": "upload"}).json()
    assert done["counts"] == {"created": 2} and done["links"]["resolved"] == 1
    stats = client.get("/api/stats").json()
    assert stats["inbox"] == 2
    assert client.get("/api/imports").json()[0]["route"] == "inbox"


def test_backup_and_exports(client, settings) -> None:
    client.post("/api/records", json={"title": "Exported", "category": "facts", "content": "body", "tags": ["t"]})
    made = client.post("/api/backup", json={"label": "test"})
    assert made.status_code == 201 and made.json()["name"].endswith("-test.db")
    assert client.get("/api/backups").json()[0]["name"] == made.json()["name"]
    assert client.get("/api/stats").json()["last_backup"] == made.json()["name"]

    exported = client.get("/api/export/json")
    assert "attachment" in exported.headers["content-disposition"]
    data = exported.json()
    assert data["format"] == "cortex-export" and data["records"][0]["title"] == "Exported"

    zipped = client.get("/api/export/markdown")
    with zipfile.ZipFile(io.BytesIO(zipped.content)) as zf:
        md = [n for n in zf.namelist() if n.startswith("facts/")]
        assert len(md) == 1
        text = zf.read(md[0]).decode("utf-8")
        assert text.startswith("---\n") and "title: Exported" in text and text.endswith("body")


def test_resolve_endpoint(client) -> None:
    client.post("/api/records", json={"title": "Ibarra 2005", "category": "facts", "source_ref": "v/03 Literature/Ibarra 2005.md"})
    assert client.get("/api/resolve", params={"ref": "03 Literature/Ibarra 2005"}).json()["title"] == "Ibarra 2005"
    assert client.get("/api/resolve", params={"ref": "missing"}).status_code == 404


def test_activity_feed(client) -> None:
    client.post("/api/records", json={"title": "One", "category": "inbox"})
    feed = client.get("/api/activity").json()
    assert feed[0]["action"] == "created" and feed[0]["title"] == "One"
    assert json.dumps(feed)  # serializable
