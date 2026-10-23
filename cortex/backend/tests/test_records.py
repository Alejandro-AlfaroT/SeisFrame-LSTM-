import json

import pytest

from cortex_server import repository as repo
from tests.conftest import make


def test_create_and_read_record_with_all_fields(conn) -> None:
    project = make(conn, title="SeisFrame V6", category="projects", kind="project", status="active")
    rid = make(
        conn,
        title="Columns are capped at 32 in.",
        content="Design rule from the grouped engine.",
        category="facts",
        kind="fact",
        project_id=project,
        source_ref="README.md#design-basis",
        confidence=0.8,
        tags=["Design", "#ACI 318", "design"],
        occurred_at="2026-10-16",
        metadata={"owner": "user"},
    )
    rec = repo.get_record(conn, rid)
    assert rec["title"] == "Columns are capped at 32 in."
    assert rec["category"] == "facts" and rec["region"] == "hippocampus"
    assert rec["project"] == {"id": project, "title": "SeisFrame V6"}
    assert rec["tags"] == ["aci-318", "design"]
    assert rec["confidence"] == 0.8 and rec["verified"] is False
    assert rec["metadata"] == {"owner": "user"}
    assert rec["created_at"] and rec["updated_at"] == rec["created_at"]


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"title": "  ", "category": "facts"}, "title"),
        ({"title": "x", "category": "dreams"}, "category"),
        ({"title": "x", "category": "facts", "confidence": 1.5}, "confidence"),
        ({"title": "x", "category": "facts", "occurred_at": "last tuesday"}, "occurred_at"),
        ({"title": "x", "category": "facts", "kind": "Not A Slug"}, "kind"),
        ({"title": "x", "category": "facts", "project_id": "missing"}, "project_id"),
        ({"title": "x", "category": "facts", "colour": "red"}, "Unknown"),
    ],
)
def test_invalid_input_is_rejected(conn, fields, message) -> None:
    with pytest.raises(repo.InvalidInput, match=message):
        repo.create_record(conn, fields)


def test_project_id_must_point_at_a_project(conn) -> None:
    fact = make(conn, title="Not a project")
    with pytest.raises(repo.InvalidInput):
        make(conn, title="child", project_id=fact)


def test_every_update_keeps_the_previous_version_and_can_be_reverted(conn) -> None:
    rid = make(conn, title="Draft", content="first")
    assert repo.update_record(conn, rid, {"content": "second", "tags": ["a"]})
    assert repo.update_record(conn, rid, {"content": "third"})
    assert repo.update_record(conn, rid, {"content": "third"}) is False  # no-op is not versioned
    versions = repo.record_versions(conn, rid)
    assert [v["snapshot"]["content"] for v in versions] == ["second", "first"]
    oldest = versions[-1]["id"]
    repo.revert_record(conn, rid, oldest)
    rec = repo.get_record(conn, rid)
    assert rec["content"] == "first" and rec["tags"] == []
    assert rec["version_count"] == 3


def test_stale_write_is_refused(conn) -> None:
    rid = make(conn, title="Shared")
    loaded = repo.get_record(conn, rid)["updated_at"]
    repo.update_record(conn, rid, {"content": "someone else saved"}, expected_updated_at=loaded)
    with pytest.raises(repo.StaleWrite):
        repo.update_record(conn, rid, {"content": "my older copy"}, expected_updated_at=loaded)
    assert repo.get_record(conn, rid)["content"] == "someone else saved"


def test_verification_sets_and_clears_timestamp(conn) -> None:
    rid = make(conn, title="Fact")
    repo.update_record(conn, rid, {"verified": True})
    assert repo.get_record(conn, rid)["verified_at"]
    repo.update_record(conn, rid, {"verified": False})
    assert repo.get_record(conn, rid)["verified_at"] is None


def test_delete_is_a_soft_trash_until_explicitly_purged(conn) -> None:
    rid = make(conn, title="Old idea", category="inbox")
    other = make(conn, title="Related")
    repo.create_relationship(conn, rid, other, "relates_to")
    with pytest.raises(repo.InvalidInput):
        repo.purge_record(conn, rid)  # not in the trash yet
    repo.trash_record(conn, rid)
    assert repo.get_record(conn, rid)["deleted"] is True
    assert repo.stats(conn)["trash"] == 1
    with pytest.raises(repo.InvalidInput):
        repo.update_record(conn, rid, {"title": "edit while trashed"})
    repo.restore_record(conn, rid)
    assert repo.get_record(conn, rid)["deleted"] is False
    repo.trash_record(conn, rid)
    repo.purge_record(conn, rid)
    with pytest.raises(repo.NotFound):
        repo.get_record(conn, rid)
    assert conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM records_fts WHERE record_id = ?", (rid,)).fetchone()[0] == 0
    actions = [a["action"] for a in repo.recent_activity(conn)]
    assert actions[:3] == ["purged", "trashed", "restored"]


def test_relationships_are_typed_unique_and_editable(conn) -> None:
    a = make(conn, title="A")
    b = make(conn, title="B")
    rel = repo.create_relationship(conn, a, b, "supports", "because")
    assert repo.create_relationship(conn, a, b, "supports") == rel  # idempotent
    with pytest.raises(repo.InvalidInput):
        repo.create_relationship(conn, a, b, "likes")
    with pytest.raises(repo.InvalidInput):
        repo.create_relationship(conn, a, a, "supports")
    repo.update_relationship(conn, rel, rel_type="contradicts")
    out = repo.relationships_for(conn, a)["outgoing"]
    assert out[0]["rel_type"] == "contradicts" and out[0]["note"] == "because"
    assert repo.relationships_for(conn, b)["incoming"][0]["other"]["title"] == "A"
    repo.delete_relationship(conn, rel)
    assert repo.relationships_for(conn, a) == {"outgoing": [], "incoming": []}


def test_stats_match_the_dashboard_definitions(conn) -> None:
    p = make(conn, title="Active project", category="projects", kind="project", status="active")
    make(conn, title="Done project", category="projects", kind="project", status="complete")
    make(conn, title="Verified", verified=True)
    make(conn, title="Unverified")
    make(conn, title="Session", category="sessions", kind="session", project_id=p)
    make(conn, title="Inbox", category="inbox")
    s = repo.stats(conn)
    assert s["total_records"] == 6
    assert s["verified_facts"] == 1 and s["unverified_facts"] == 1
    assert s["active_projects"] == 1
    assert s["sessions"] == 1 and s["inbox"] == 1


def test_control_characters_are_stripped_from_text(conn) -> None:
    rid = make(conn, title="Bell\x07 title", content="null\x00byte")
    rec = repo.get_record(conn, rid)
    assert rec["title"] == "Bell title" and rec["content"] == "nullbyte"


def test_resolve_reference_by_path_and_title(conn) -> None:
    rid = make(conn, title="Ibarra 2005", source_ref="vault/03 Literature/Ibarra 2005.md")
    assert repo.resolve_reference(conn, "03 Literature/Ibarra 2005")["id"] == rid
    assert repo.resolve_reference(conn, "Ibarra 2005#Section")["id"] == rid
    assert repo.resolve_reference(conn, "nothing here") is None


def test_metadata_is_stored_as_json(conn) -> None:
    rid = make(conn, title="m", metadata={"nested": {"a": [1, 2]}})
    raw = conn.execute("SELECT metadata FROM records WHERE id = ?", (rid,)).fetchone()[0]
    assert json.loads(raw) == {"nested": {"a": [1, 2]}}
