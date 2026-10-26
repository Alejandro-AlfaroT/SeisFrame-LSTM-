import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cortex_server import conflicts, documents
from cortex_server import repository as repo
from cortex_server.importer import ImportOptions, PdfSource, collect_directory, import_markdown
from cortex_server.search import SNIPPET_OPEN, Filters, search
from tests.pdfs import make_pdf

IBARRA = [
    "Hysteretic models that incorporate strength and stiff-\nness deterioration",
    None,  # a full-page figure: no text
    "The cyclic deterioration parameter Lambda controls energy dissipation",
]


def vault(tmp_path: Path, pages: list[str | None] = IBARRA) -> Path:
    root = tmp_path / "Vault"
    (root / "Papers").mkdir(parents=True)
    (root / "Assets").mkdir()
    (root / "Papers" / "Ibarra 2005.md").write_text(
        "---\ntype: literature\n---\n# Ibarra 2005\n\nSource: [[Assets/Ibarra 2005.pdf|IMK2005Original.pdf]]\n",
        encoding="utf-8",
    )
    (root / "Assets" / "Ibarra 2005.pdf").write_bytes(make_pdf(pages))
    return root


def run_import(conn: sqlite3.Connection, root: Path, **kw):  # type: ignore[no-untyped-def]
    pdfs: list[PdfSource] = []
    files, skipped = collect_directory(root, pdfs=pdfs)
    options = ImportOptions(route="by_type", source_label=root.name, root=str(root), **kw)
    return import_markdown(conn, files, options, skipped, pdfs)


def pdf_record(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute("SELECT * FROM records WHERE kind = 'pdf'").fetchone()


def test_clean_page_text_rejoins_broken_words_and_folds_ligatures() -> None:
    raw = "The ﬁrst defor-\nmation  test\t of a\n\n\n\nbeam-column joint\r\n"
    assert documents.clean_page_text(raw) == "The first deformation test of a\n\nbeam-column joint"


def test_extraction_reads_every_page_in_a_separate_process(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(IBARRA))
    result = documents.extract_pdf(path)
    assert result.status == "ok" and result.page_count == 3 and result.text_pages == 2
    assert result.pages[1] == "Hysteretic models that incorporate strength and stiffness deterioration"
    assert 2 not in result.pages


def test_scanned_pdf_has_no_text(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf([None, None]))
    result = documents.extract_pdf(path)
    assert (result.status, result.page_count, result.pages) == ("no_text", 2, {})


def test_broken_pdf_is_reported_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not really a PDF\n")
    result = documents.extract_pdf(path)
    assert result.status == "failed" and result.error


def test_extraction_is_stopped_after_the_time_limit(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(IBARRA))
    result = documents.extract_pdf(path, timeout=0.001)
    assert result.status == "failed" and "stopped after" in (result.error or "")


def test_password_protected_pdf_is_not_opened_but_owner_locked_ones_are(tmp_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    plain = tmp_path / "plain.pdf"
    plain.write_bytes(make_pdf(IBARRA))
    for name, user_password in (("locked.pdf", "secret"), ("owner-only.pdf", "")):
        writer = PdfWriter(clone_from=PdfReader(plain))
        writer.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-128")
        with (tmp_path / name).open("wb") as f:
            writer.write(f)
    locked = documents.extract_pdf(tmp_path / "locked.pdf")
    assert locked.status == "failed" and locked.error == "password-protected"
    assert documents.extract_pdf(tmp_path / "owner-only.pdf").text_pages == 2


def test_import_makes_a_record_per_pdf_and_resolves_links_to_it(conn: sqlite3.Connection, tmp_path: Path) -> None:
    report = run_import(conn, vault(tmp_path))
    assert report["counts"] == {"created": 2}
    assert report["pdfs"]["found"] == 1 and report["pdfs"]["statuses"] == {"ok": 1}
    pdf = pdf_record(conn)
    assert (pdf["title"], pdf["category"], pdf["source_ref"]) == ("Ibarra 2005", "facts", "Vault/Assets/Ibarra 2005.pdf")
    assert "3 pages" in pdf["content"]
    note = conn.execute("SELECT id FROM records WHERE kind = 'literature'").fetchone()
    links = repo.relationships_for(conn, note["id"])["outgoing"]
    assert [(link["rel_type"], link["other"]["id"]) for link in links] == [("references", pdf["id"])]
    assert report["links"] == {"resolved": 1, "unresolved": 0}
    record = repo.get_record(conn, pdf["id"])
    assert record["document"]["page_count"] == 3 and record["document"]["text_pages"] == 2
    assert repo.resolve_reference(conn, "Assets/Ibarra 2005.pdf")["id"] == pdf["id"]
    assert repo.resolve_reference(conn, "Ibarra 2005")["id"] == note["id"]  # the note, not the PDF


def test_search_finds_passages_inside_pdfs(conn: sqlite3.Connection, tmp_path: Path) -> None:
    run_import(conn, vault(tmp_path))
    result = search(conn, "stiffness", Filters())
    docs = result["documents"]
    assert docs["total_documents"] == 1 and docs["total_pages"] == 1
    hit = docs["results"][0]
    assert hit["title"] == "Ibarra 2005" and hit["page_count"] == 3 and hit["pages_matched"] == 1
    assert hit["passages"][0]["page"] == 1
    assert f"{SNIPPET_OPEN}stiffness" in hit["passages"][0]["snippet"]
    assert search(conn, "deteriorate", Filters())["documents"]["total_pages"] == 2  # stemmed
    assert search(conn, '"energy dissipation"', Filters())["documents"]["results"][0]["passages"][0]["page"] == 3
    assert search(conn, "stiffness", Filters(categories=["projects"]))["documents"]["total_documents"] == 0


def test_similar_records_compare_a_pdf_by_its_opening_pages(conn: sqlite3.Connection, tmp_path: Path) -> None:
    from cortex_server.search import similar

    root = vault(tmp_path)
    (root / "Assets" / "Other report.pdf").write_bytes(make_pdf(["Wind tunnel tests of a tall chimney"]))
    run_import(conn, root)
    found = similar(conn, text="cyclic deterioration of stiffness and energy dissipation")
    assert [(r["title"], r["kind"]) for r in found["results"][:1]] == [("Ibarra 2005", "pdf")]
    assert "Other report" not in [r["title"] for r in found["results"]]


def test_page_text_marks_matches_and_lists_matching_pages(conn: sqlite3.Connection, tmp_path: Path) -> None:
    run_import(conn, vault(tmp_path))
    pdf_id = pdf_record(conn)["id"]
    page = documents.page_text(conn, pdf_id, 3, "deterioration")
    assert page["matches"] == [1, 3] and page["page_count"] == 3
    assert f"{SNIPPET_OPEN}deterioration" in page["text"]
    blank = documents.page_text(conn, pdf_id, 2)
    assert blank["has_text"] is False and blank["text"] == ""
    with pytest.raises(repo.NotFound):
        documents.page_text(conn, pdf_id, 4)


def test_reimport_reads_only_changed_pdfs(conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = vault(tmp_path)
    run_import(conn, root)
    pdf_id = pdf_record(conn)["id"]

    def refuse(*_args, **_kw):  # type: ignore[no-untyped-def]
        raise AssertionError("an unchanged PDF was read again")

    monkeypatch.setattr(documents, "extract_pdf", refuse)
    again = run_import(conn, root)
    assert again["counts"] == {"unchanged": 2} and again["pdfs"]["read"] == 0
    monkeypatch.undo()

    (root / "Assets" / "Ibarra 2005.pdf").write_bytes(make_pdf(["Revised edition about plastic rotation capacity"]))
    changed = run_import(conn, root)
    assert changed["counts"] == {"updated": 1, "unchanged": 1}
    assert pdf_record(conn)["id"] == pdf_id
    assert search(conn, "stiffness", Filters())["documents"]["total_pages"] == 0  # old text is gone
    assert search(conn, "rotation", Filters())["documents"]["results"][0]["id"] == pdf_id
    assert repo.get_record(conn, pdf_id)["document"]["page_count"] == 1


def test_pdf_record_edited_in_cortex_keeps_its_text_but_gets_new_pages(conn: sqlite3.Connection, tmp_path: Path) -> None:
    root = vault(tmp_path)
    run_import(conn, root)
    pdf_id = pdf_record(conn)["id"]
    repo.update_record(conn, pdf_id, {"content": "My reading notes on Ibarra."})
    (root / "Assets" / "Ibarra 2005.pdf").write_bytes(make_pdf(["Second edition on plastic rotation"]))
    report = run_import(conn, root)
    entry = next(e for e in report["files"] if e["kind"] == "pdf")
    assert entry["action"] == "updated" and entry["warnings"]
    assert pdf_record(conn)["content"] == "My reading notes on Ibarra."
    assert search(conn, "rotation", Filters())["documents"]["total_documents"] == 1


def test_dry_run_reads_no_pdf_and_writes_nothing(conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents, "extract_pdf", lambda *a, **k: pytest.fail("dry run read a PDF"))
    report = run_import(conn, vault(tmp_path), dry_run=True)
    assert report["counts"] == {"created": 2}
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_without_pypdf_pdfs_are_skipped_and_notes_still_import(
    conn: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(documents, "extractor_id", lambda: None)
    report = run_import(conn, vault(tmp_path))
    assert report["counts"] == {"created": 1}
    assert {"path": "Assets/Ibarra 2005.pdf", "reason": "reading PDFs needs the pypdf package"} in report["skipped"]


def test_broken_and_scanned_pdfs_get_records_that_say_why(conn: sqlite3.Connection, tmp_path: Path) -> None:
    root = vault(tmp_path)
    (root / "Assets" / "Scan.pdf").write_bytes(make_pdf([None]))
    (root / "Assets" / "Broken.pdf").write_bytes(b"%PDF-1.4\nnot a pdf")
    report = run_import(conn, root)
    statuses = {a["path"]: a["status"] for a in report["pdfs"]["attention"]}
    assert statuses == {"Assets/Broken.pdf": "failed", "Assets/Scan.pdf": "no_text"}
    scan = conn.execute("SELECT content FROM records WHERE title = 'Scan'").fetchone()[0]
    assert "does not run OCR" in scan
    broken = conn.execute("SELECT content FROM records WHERE title = 'Broken'").fetchone()[0]
    assert "could not be extracted" in broken


def test_trash_hides_passages_and_purge_removes_the_text(conn: sqlite3.Connection, tmp_path: Path) -> None:
    run_import(conn, vault(tmp_path))
    pdf_id = pdf_record(conn)["id"]
    repo.trash_record(conn, pdf_id)
    assert search(conn, "stiffness", Filters())["documents"]["total_pages"] == 0
    repo.purge_record(conn, pdf_id)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM document_pages").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM document_pages_fts WHERE document_pages_fts MATCH 'stiffness'").fetchone()[0] == 0
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_conflict_detection_ignores_generated_pdf_records(conn: sqlite3.Connection, tmp_path: Path) -> None:
    root = vault(tmp_path)
    for name, n in (("A", 3), ("B", 5), ("C", 7), ("D", 9)):
        (root / "Assets" / f"{name}.pdf").write_bytes(make_pdf([f"Report {name} page {i}" for i in range(n)]))
    run_import(conn, root)
    assert conflicts.detect(conn)["new_conflicts"] == []


def test_api_serves_pages_and_the_original_file(conn: sqlite3.Connection, client: TestClient, tmp_path: Path) -> None:
    root = vault(tmp_path)
    run_import(conn, root)
    pdf_id = pdf_record(conn)["id"]
    note_id = conn.execute("SELECT id FROM records WHERE kind = 'literature'").fetchone()[0]

    found = client.get("/api/search", params={"q": "stiffness"}).json()
    assert found["documents"]["results"][0]["passages"][0]["page"] == 1
    page = client.get(f"/api/records/{pdf_id}/pages/1", params={"q": "stiffness"}).json()
    assert page["matches"] == [1] and SNIPPET_OPEN in page["text"]
    assert client.get(f"/api/records/{pdf_id}/pages/9").status_code == 404
    assert client.get(f"/api/records/{pdf_id}").json()["document"]["status"] == "ok"
    assert client.get("/api/stats").json()["pdfs"] == 1

    response = client.get(f"/api/records/{pdf_id}/file")
    assert response.status_code == 200 and response.content.startswith(b"%PDF")
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline")
    assert client.get(f"/api/records/{note_id}/file").status_code == 404

    (root / "Assets" / "Ibarra 2005.pdf").unlink()
    missing = client.get(f"/api/records/{pdf_id}/file")
    assert missing.status_code == 404 and "no longer at" in missing.json()["detail"]


def test_json_export_lists_pdfs_but_not_their_page_text(conn: sqlite3.Connection, tmp_path: Path) -> None:
    from cortex_server.backup import export_json

    run_import(conn, vault(tmp_path))
    data = export_json(conn)
    assert len(data["documents"]) == 1 and data["documents"][0]["pdf_meta"] == {}
    assert "document_pages" in data["omitted"] and "document_pages" not in data
