"""Small, valid PDFs for tests, written byte by byte (no PDF library involved)."""

from __future__ import annotations


def make_pdf(pages: list[str | None]) -> bytes:
    """One page per entry. A string becomes that page's text (one line per \\n); None makes a
    page that has only a filled shape and no text, like a scanned image."""
    page_ids = [4 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for page_id, text in zip(page_ids, pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>"
            f" /Contents {page_id + 1} 0 R >>".encode()
        )
        if text is None:
            stream = b"0.5 g 72 72 468 648 re f"
        else:
            lines = []
            for row, line in enumerate(text.split("\n")):
                escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                lines.append(f"BT /F1 11 Tf 72 {720 - 14 * row} Td ({escaped}) Tj ET")
            stream = "\n".join(lines).encode("cp1252")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
