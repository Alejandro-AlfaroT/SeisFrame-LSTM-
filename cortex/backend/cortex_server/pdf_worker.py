"""Extract the text of one PDF and print it as JSON on stdout.

Run as a separate process by documents.py (`python -I pdf_worker.py <file> <max pages>
<max chars per page>`), so a malformed or hostile PDF can only stall or crash this process,
which the parent kills after a timeout. Uses only the standard library and pypdf; it never
writes files and never runs anything embedded in the PDF.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from typing import Any


def _meta(reader: Any) -> dict[str, str]:
    try:
        info = reader.metadata or {}
    except Exception:  # noqa: BLE001 - metadata is optional
        return {}
    out = {}
    for key in ("/Title", "/Author", "/Subject", "/Keywords", "/CreationDate"):
        value = info.get(key)
        if value:
            out[key[1:].lower()] = str(value)[:500]
    return out


def extract(path: str, max_pages: int, max_chars: int) -> dict[str, Any]:
    import pypdf
    from pypdf import PdfReader

    # Larger than pypdf's defaults so long reports and figure-heavy papers parse; the parent
    # process still bounds the total time.
    pypdf.overwrite_configuration(
        zlib_maximum_output_length=300_000_000,
        stream_parsing_work_maximum_count=20_000_000,
    )
    result: dict[str, Any] = {"pypdf": pypdf.__version__, "encrypted": False}
    reader = PdfReader(path)
    if reader.is_encrypted:
        result["encrypted"] = True
        # Many published PDFs are encrypted only to restrict printing/copying and open with an
        # empty password; anything that needs a real password is reported, not guessed.
        if not reader.decrypt(""):
            return {**result, "error": "password-protected"}
    result["meta"] = _meta(reader)
    page_count = len(reader.pages)
    result["page_count"] = page_count
    pages: list[str | None] = []
    page_errors: dict[int, str] = {}
    for index in range(min(page_count, max_pages)):
        try:
            text = reader.pages[index].extract_text() or ""
            pages.append(text[:max_chars])
        except Exception as exc:  # noqa: BLE001 - one bad page must not lose the others
            pages.append(None)
            page_errors[index + 1] = f"{type(exc).__name__}: {exc}"[:200]
    result["pages"] = pages
    result["page_errors"] = page_errors
    result["truncated"] = page_count > max_pages
    return result


def main() -> int:
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    path, max_pages, max_chars = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    try:
        result = extract(path, max_pages, max_chars)
    except ImportError as exc:
        result = {"error": f"missing package: {exc.name or exc}"}
    except Exception as exc:  # noqa: BLE001 - reported to the parent as a failed document
        name = type(exc).__name__
        if name == "DependencyError":
            result = {"error": f"{exc} (re-run setup to install it)"}
        else:
            result = {"error": f"{name}: {exc}"[:300]}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
