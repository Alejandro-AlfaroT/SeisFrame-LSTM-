import os
import shutil
from pathlib import Path

import pytest

from cortex_server import importer
from cortex_server import repository as repo
from cortex_server.importer import ImportOptions, SourceFile, collect_directory, import_markdown, parse_frontmatter
from tests.conftest import FIXTURES

VAULT = FIXTURES / "vault"


def run(conn, route="by_type", dry_run=False, root=VAULT, **kw):
    files, skipped = collect_directory(root)
    return import_markdown(conn, files, ImportOptions(route=route, source_label="vault", dry_run=dry_run, **kw), skipped)


def by_title(conn, title):
    row = conn.execute("SELECT id FROM records WHERE title = ?", (title,)).fetchone()
    return repo.get_record(conn, row[0]) if row else None


def test_collect_skips_hidden_templates_and_non_markdown() -> None:
    files, skipped = collect_directory(VAULT)
    paths = sorted(f.path for f in files)
    assert paths == [
        "00 Home/Start Here.md",
        "02 Research/Bad Frontmatter.md",
        "03 Literature/Ibarra 2005.md",
        "06 Decisions/DEC-001.md",
        "07 Daily/2026-10-21.md",
    ]
    assert {"path": "Templates/", "reason": "excluded folder"} in skipped


def test_dry_run_reports_but_writes_nothing(conn) -> None:
    report = run(conn, dry_run=True)
    assert report["dry_run"] is True and report["batch_id"] is None
    assert report["counts"] == {"created": 5}
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM import_batches").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM activity").fetchone()[0] == 0


def test_by_type_routing_titles_dates_tags_and_status(conn) -> None:
    run(conn)
    decision = by_title(conn, "DEC-001 · Use paired horizontal ground motions")
    assert (decision["category"], decision["kind"], decision["status"]) == ("projects", "decision", "accepted")
    assert set(decision["tags"]) == {"imported", "decisions", "scope", "imk"}
    assert decision["source_ref"] == "vault/06 Decisions/DEC-001.md"
    assert decision["verified"] is False
    daily = by_title(conn, "Wednesday, October 21")
    assert (daily["category"], daily["kind"], daily["occurred_at"]) == ("sessions", "session", "2026-10-21")
    lit = by_title(conn, "Ibarra 2005 · Hysteretic models with deterioration")
    assert (lit["category"], lit["kind"], lit["occurred_at"]) == ("facts", "literature", "2026-10-20")
    home = by_title(conn, "Start Here")
    assert home["category"] == "inbox"


def test_inbox_route_keeps_a_suggestion_for_triage(conn) -> None:
    run(conn, route="inbox")
    decision = by_title(conn, "DEC-001 · Use paired horizontal ground motions")
    assert decision["category"] == "inbox" and decision["kind"] == "decision"
    assert decision["metadata"]["suggested_category"] == "projects"


def test_wikilinks_become_reference_relationships(conn) -> None:
    report = run(conn)
    home = by_title(conn, "Start Here")
    targets = {r["other"]["title"] for r in home["relationships"]["outgoing"]}
    assert targets == {"DEC-001 · Use paired horizontal ground motions", "Wednesday, October 21"}
    assert all(r["rel_type"] == "references" and r["origin"] == "import" for r in home["relationships"]["outgoing"])
    assert home["metadata"]["import"]["unresolved_links"] == ["Nowhere/Missing Note"]
    daily = by_title(conn, "Wednesday, October 21")  # [[DEC-001]] resolves by unique file name
    assert [r["other"]["title"] for r in daily["relationships"]["outgoing"]] == ["DEC-001 · Use paired horizontal ground motions"]
    assert report["links"] == {"resolved": 4, "unresolved": 1}


def test_untrusted_content_is_stored_verbatim_and_bad_yaml_is_ignored(conn) -> None:
    report = run(conn)
    lit = by_title(conn, "Ibarra 2005 · Hysteretic models with deterioration")
    assert '<script>alert("never executed")</script>' in lit["content"]
    bad = next(e for e in report["files"] if e["path"] == "02 Research/Bad Frontmatter.md")
    assert bad["warnings"] == ["frontmatter is not valid YAML; ignored"]
    assert by_title(conn, "Bad Frontmatter")["content"].strip() == "Body after broken YAML."


def test_yaml_cannot_construct_python_objects() -> None:
    fm, body, warnings = parse_frontmatter("---\nx: !!python/object/apply:os.system ['echo pwned']\n---\nbody")
    assert fm == {} and warnings == ["frontmatter is not valid YAML; ignored"] and body == "body"


def test_reimport_unchanged_updated_local_edits_and_trash(conn, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    shutil.copytree(VAULT, vault)
    first = run(conn, root=vault)
    assert first["counts"] == {"created": 5}
    assert run(conn, root=vault)["counts"] == {"unchanged": 5}

    (vault / "07 Daily" / "2026-10-21.md").write_text("---\ntype: daily\n---\n# Wednesday, October 21\n\nRevised.\n", "utf-8")
    edited_inside = by_title(conn, "Start Here")
    repo.update_record(conn, edited_inside["id"], {"content": "my own notes"})
    (vault / "00 Home" / "Start Here.md").write_text("# Start Here\n\nChanged on disk.\n", "utf-8")
    lit = by_title(conn, "Ibarra 2005 · Hysteretic models with deterioration")
    repo.trash_record(conn, lit["id"])
    (vault / "03 Literature" / "Ibarra 2005.md").write_text("# Ibarra 2005\n\nchanged\n", "utf-8")

    report = run(conn, root=vault)
    actions = {e["path"]: e["action"] for e in report["files"]}
    assert actions["07 Daily/2026-10-21.md"] == "updated"
    assert actions["00 Home/Start Here.md"] == "skipped-local-edits"
    assert actions["03 Literature/Ibarra 2005.md"] == "skipped-trashed"
    daily = by_title(conn, "Wednesday, October 21")
    assert "Revised." in daily["content"]
    assert repo.record_versions(conn, daily["id"])[0]["snapshot"]["content"].strip().startswith("# Wednesday")
    assert by_title(conn, "Start Here")["content"] == "my own notes"

    forced = run(conn, root=vault, update_edited=True)
    assert {e["path"]: e["action"] for e in forced["files"]}["00 Home/Start Here.md"] == "updated"
    restored_history = repo.record_versions(conn, edited_inside["id"])
    assert restored_history[0]["snapshot"]["content"] == "my own notes"  # nothing lost


def test_triaged_category_survives_reimport(conn, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    shutil.copytree(VAULT, vault)
    run(conn, root=vault, route="inbox", update_edited=True)
    rec = by_title(conn, "DEC-001 · Use paired horizontal ground motions")
    repo.update_record(conn, rec["id"], {"category": "projects"})
    (vault / "06 Decisions" / "DEC-001.md").write_text("---\ntype: decision\n---\n# DEC-001 · Use paired horizontal ground motions\n\nv2\n", "utf-8")
    run(conn, root=vault, route="inbox", update_edited=True)
    after = repo.get_record(conn, rec["id"])
    assert after["category"] == "projects" and "v2" in after["content"]


@pytest.mark.parametrize(
    "path, reason",
    [
        ("../outside.md", "must not contain '..'"),
        ("/etc/passwd.md", "path must be relative"),
        ("C:/Windows/x.md", "path must be relative"),
        ("notes/script.js", "not a Markdown file"),
        (".obsidian/a.md", "hidden folder"),
        ("Templates/t.md", "excluded folder"),
    ],
)
def test_upload_paths_are_validated(path, reason) -> None:
    norm, why = importer.validate_upload(path, "# x", importer.DEFAULT_EXCLUDED_DIRS)
    assert norm is None and reason in why


def test_upload_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(importer, "MAX_FILE_BYTES", 10)
    assert importer.validate_upload("a.md", "x" * 11, ())[0] is None
    assert importer.validate_upload("a.md", "x" * 10, ())[0] == "a.md"


def test_large_files_on_disk_are_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(importer, "MAX_FILE_BYTES", 10)
    (tmp_path / "big.md").write_text("x" * 50, "utf-8")
    (tmp_path / "ok.md").write_text("small", "utf-8")
    files, skipped = collect_directory(tmp_path)
    assert [f.path for f in files] == ["ok.md"]
    assert skipped[0]["path"] == "big.md"


def test_symlinks_are_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", "utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.md").write_text("real", "utf-8")
    try:
        os.symlink(outside / "secret.md", root / "link.md")
        os.symlink(outside, root / "linked_dir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs Developer Mode or admin rights on this machine")
    files, skipped = collect_directory(root)
    assert [f.path for f in files] == ["real.md"]
    assert {s["path"] for s in skipped} == {"link.md", "linked_dir/"}


def test_project_assignment_and_new_conflicts_are_reported(conn) -> None:
    project = repo.create_record(conn, {"title": "SeisFrame V6", "category": "projects", "kind": "project"})
    files = [
        SourceFile("a.md", "---\ntype: fact\n---\n# A\n\nThe no-slab branch retains slab mass and diaphragm constraints.\n"),
        SourceFile("b.md", "---\ntype: fact\n---\n# B\n\nThe no-slab branch removes slab mass and diaphragm constraints.\n"),
    ]
    report = import_markdown(conn, files, ImportOptions(route="by_type", project_id=project, source_label="x"))
    assert report["conflicts"] == {"new": 1}
    assert all(by_title(conn, t)["project_id"] == project for t in ("A", "B"))
