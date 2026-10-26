"""PDF documents: text extraction, storage of page text, and page lookup.

PDFs are untrusted input. Each one is parsed by pypdf in a child process (pdf_worker.py) that is
killed after PDF_TIMEOUT_SECONDS, so a malformed file can stall or crash only that process.
Nothing in a PDF is executed; only its text and a few metadata fields are kept, and the file
itself is never copied or changed. Pages that are only images (scans) have no text layer:
CORTEX does not run OCR, so such documents are stored with status `no_text`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from .db import utcnow
from .repository import NotFound
from .search import SNIPPET_CLOSE, SNIPPET_OPEN, build_match_query
from .textutil import sanitize_text

MAX_PDF_BYTES = 512 * 1024 * 1024
MAX_PAGES = 5000
MAX_PAGE_CHARS = 200_000
MIN_TEXT_CHARS = 20  # a page with fewer non-space characters (e.g. just a page number) has no text
PDF_TIMEOUT_SECONDS = float(os.environ.get("CORTEX_PDF_TIMEOUT", "300"))
PIPELINE_VERSION = 1  # bump when clean_page_text changes, so stored text is extracted again
WORKER = Path(__file__).with_name("pdf_worker.py")
META_KEYS = ("title", "author", "subject", "keywords", "creationdate")


def extractor_id() -> str | None:
    """Identifies the extraction pipeline; None when pypdf is not installed."""
    try:
        major = metadata.version("pypdf").split(".")[0]
    except metadata.PackageNotFoundError:
        return None
    return f"pypdf-{major}/cortex-{PIPELINE_VERSION}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- extraction

_HYPHEN_BREAK_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_page_text(text: str) -> str:
    """Searchable text: ligatures and compatibility forms folded (NFKC), words broken across
    lines re-joined, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub("", sanitize_text(text))
    text = "\n".join(_SPACES_RE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


@dataclass
class Extraction:
    status: str  # ok | partial | no_text | failed
    page_count: int = 0
    pages: dict[int, str] = field(default_factory=dict)  # 1-based page number -> text
    text_pages: int = 0
    failed_pages: int = 0
    error: str | None = None
    meta: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


def _from_worker(data: dict[str, Any], seconds: float) -> Extraction:
    meta = {
        key: sanitize_text(str(value))[:500]
        for key, value in (data.get("meta") or {}).items()
        if key in META_KEYS and value
    }
    page_count = int(data.get("page_count") or 0)
    if data.get("error"):
        return Extraction("failed", page_count, error=sanitize_text(str(data["error"]))[:300], meta=meta, seconds=seconds)
    pages: dict[int, str] = {}
    text_pages = 0
    for number, raw in enumerate(data.get("pages") or [], start=1):
        if not isinstance(raw, str):
            continue
        text = clean_page_text(raw)[:MAX_PAGE_CHARS]
        if text:
            pages[number] = text
            if len(re.sub(r"\s", "", text)) >= MIN_TEXT_CHARS:
                text_pages += 1
    page_errors = data.get("page_errors") or {}
    failed = len(page_errors)
    notes = []
    if failed:
        notes.append(f"{failed} page{'s' if failed != 1 else ''} could not be read")
    if data.get("truncated"):
        notes.append(f"only the first {MAX_PAGES} pages were read")
    if page_count and failed >= page_count:
        status = "failed"
        notes.insert(0, str(next(iter(page_errors.values()))))
    elif text_pages == 0:
        status = "no_text"
    elif notes or text_pages < page_count / 2:
        status = "partial"
    else:
        status = "ok"
    return Extraction(status, page_count, pages, text_pages, failed, "; ".join(notes) or None, meta, seconds)


def extract_pdf(path: Path, timeout: float | None = None) -> Extraction:
    """Text of every page of one PDF, read in a separate, time-limited process."""
    limit = timeout if timeout is not None else PDF_TIMEOUT_SECONDS
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(WORKER), str(path), str(MAX_PAGES), str(MAX_PAGE_CHARS)],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=limit,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return Extraction("failed", error=f"stopped after {limit:g} s", seconds=time.perf_counter() - started)
    seconds = time.perf_counter() - started
    try:
        data = json.loads(proc.stdout.decode("utf-8"))
    except ValueError:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = f": {lines[-1][:200]}" if lines else ""
        return Extraction("failed", error=f"extractor exited with code {proc.returncode}{detail}", seconds=seconds)
    return _from_worker(data, seconds)


def extract_many(
    paths: list[Path],
    progress: Callable[[int, int, Path, Extraction], None] | None = None,
    workers: int | None = None,
) -> dict[Path, Extraction]:
    """Extract several PDFs, a few at a time. `progress(done, total, path, result)` is called as
    each one finishes."""
    results: dict[Path, Extraction] = {}
    if not paths:
        return results
    workers = workers or max(1, min(3, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_pdf, path): path for path in paths}
        for done, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            results[path] = future.result()
            if progress:
                progress(done, len(paths), path, results[path])
    return results


# ---------------------------------------------------------------- storage


def stored(conn: sqlite3.Connection, record_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE record_id = ?", (record_id,)).fetchone()


def store(
    conn: sqlite3.Connection,
    record_id: str,
    path: Path,
    sha256: str,
    size: int,
    mtime_ns: int,
    extraction: Extraction,
    extractor: str,
) -> None:
    """Save (or replace) the extracted text of a record's PDF. Call inside a transaction."""
    conn.execute(
        "INSERT INTO documents (record_id, path, sha256, bytes, mtime_ns, page_count, text_pages, failed_pages,"
        " status, error, pdf_meta, extractor, extracted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (record_id) DO UPDATE SET path = excluded.path, sha256 = excluded.sha256,"
        " bytes = excluded.bytes, mtime_ns = excluded.mtime_ns, page_count = excluded.page_count,"
        " text_pages = excluded.text_pages, failed_pages = excluded.failed_pages, status = excluded.status,"
        " error = excluded.error, pdf_meta = excluded.pdf_meta, extractor = excluded.extractor,"
        " extracted_at = excluded.extracted_at",
        (
            record_id, str(path), sha256, size, mtime_ns, extraction.page_count, extraction.text_pages,
            extraction.failed_pages, extraction.status, extraction.error,
            json.dumps(extraction.meta, ensure_ascii=False), extractor, utcnow(),
        ),
    )
    conn.execute("DELETE FROM document_pages WHERE record_id = ?", (record_id,))
    conn.executemany(
        "INSERT INTO document_pages (record_id, page, text) VALUES (?, ?, ?)",
        [(record_id, page, text) for page, text in sorted(extraction.pages.items())],
    )


def note_location(conn: sqlite3.Connection, record_id: str, path: Path, mtime_ns: int) -> None:
    """The same file (same hash) was found again, possibly moved or touched."""
    conn.execute("UPDATE documents SET path = ?, mtime_ns = ? WHERE record_id = ?", (str(path), mtime_ns, record_id))


def describe(extraction: Extraction, rel_path: str) -> str:
    """Generated Markdown body of a PDF's record."""
    n = extraction.page_count
    pages = f"{n} page{'s' if n != 1 else ''}"
    if extraction.status == "ok":
        state = f"PDF · {pages} · the text of every page is searchable in CORTEX."
    elif extraction.status == "partial":
        state = f"PDF · {pages} · text found on {extraction.text_pages} of them and searchable in CORTEX."
        if extraction.error:
            state += f" Note: {extraction.error}."
        elif extraction.text_pages < n / 2:
            state += " Most pages are images (figures or scans), which have no text to search."
    elif extraction.status == "no_text":
        state = (
            f"PDF · {pages} · no text layer: the pages are images, as in a scan. CORTEX does not run"
            " OCR, so the contents of this PDF are not searchable; its title, tags and links are."
        )
    else:
        state = f"PDF · the text could not be extracted ({extraction.error or 'unknown error'})."
    return f"{state}\n\nFile: `{rel_path}`\n"


# ---------------------------------------------------------------- reading


def page_text(conn: sqlite3.Connection, record_id: str, page: int, q: str | None = None) -> dict[str, Any]:
    """One page of a PDF's text. With `q`, matching words are marked and `matches` lists every
    page of this PDF that matches."""
    doc = stored(conn, record_id)
    if doc is None:
        raise NotFound("this record has no PDF text")
    if not 1 <= page <= max(doc["page_count"], 1):
        raise NotFound(f"page {page} is outside 1–{doc['page_count']}")
    row = conn.execute(
        "SELECT id, text FROM document_pages WHERE record_id = ? AND page = ?", (record_id, page)
    ).fetchone()
    text = row["text"] if row else ""
    matches: list[int] = []
    match = build_match_query(q or "")
    if match:
        try:
            matches = [
                r[0]
                for r in conn.execute(
                    "SELECT p.page FROM document_pages_fts JOIN document_pages p ON p.id = document_pages_fts.rowid"
                    " WHERE document_pages_fts MATCH ? AND p.record_id = ? ORDER BY p.page",
                    (match, record_id),
                )
            ]
            if row is not None and page in matches:
                marked = conn.execute(
                    f"SELECT highlight(document_pages_fts, 0, '{SNIPPET_OPEN}', '{SNIPPET_CLOSE}')"
                    " FROM document_pages_fts WHERE document_pages_fts MATCH ? AND rowid = ?",
                    (match, row["id"]),
                ).fetchone()
                text = marked[0] if marked else text
        except sqlite3.OperationalError:
            matches = []
    return {
        "record_id": record_id,
        "page": page,
        "page_count": doc["page_count"],
        "has_text": row is not None,
        "text": text,
        "matches": matches,
    }


def file_path(conn: sqlite3.Connection, record_id: str) -> Path:
    """The original PDF on disk, if it is still where it was imported from."""
    doc = stored(conn, record_id)
    if doc is None:
        raise NotFound("this record has no PDF")
    path = Path(doc["path"])
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise NotFound(f"the PDF is no longer at {path}")
    with path.open("rb") as f:
        if b"%PDF-" not in f.read(1024):
            raise NotFound(f"{path} is no longer a PDF")
    return path
