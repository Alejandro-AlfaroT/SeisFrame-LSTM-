"""Full-text search (SQLite FTS5) with filters, passage search inside PDFs, plus TF-IDF similarity.

The similarity search is lexical (TF-IDF cosine over words), computed locally. It is not a
neural-embedding semantic search; see README for how one could be plugged in.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .repository import CATEGORIES, summarize
from .textutil import index_tokens, plain_text

SNIPPET_OPEN = "\u0002"
SNIPPET_CLOSE = "\u0003"

_TERM_RE = re.compile(r'"([^"]*)"|(-?)([^\s"]+)')


def build_match_query(q: str) -> str | None:
    """User text -> safe FTS5 MATCH expression.

    Bare words become prefix terms, "quoted text" becomes a phrase, -word excludes. Every
    token is re-quoted, so FTS5 operators typed by the user cannot break the query.
    """
    positives: list[str] = []
    negatives: list[str] = []
    for phrase, negate, term in _TERM_RE.findall(q or ""):
        if phrase:
            words = re.findall(r"\w+", phrase)
            if words:
                positives.append('"' + " ".join(words) + '"')
            continue
        words = re.findall(r"\w+", term)
        if not words:
            continue
        token = '"' + " ".join(words) + '"*'
        (negatives if negate else positives).append(token)
    if not positives:
        return None
    query = " AND ".join(positives)
    for token in negatives:
        query += f" NOT {token}"
    return query


@dataclass
class Filters:
    categories: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    project_id: str | None = None
    tags: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    verified: bool | None = None
    include_demo: bool = True
    status: str | None = None

    def sql(self) -> tuple[list[str], list[Any]]:
        where = ["r.deleted_at IS NULL"]
        params: list[Any] = []
        cats = [c for c in self.categories if c in CATEGORIES]
        if cats:
            where.append(f"r.category IN ({','.join('?' * len(cats))})")
            params += cats
        if self.kinds:
            where.append(f"r.kind IN ({','.join('?' * len(self.kinds))})")
            params += self.kinds
        if self.project_id:
            where.append("(r.project_id = ? OR r.id = ?)")
            params += [self.project_id, self.project_id]
        for tag in self.tags:
            where.append("EXISTS (SELECT 1 FROM tags t WHERE t.record_id = r.id AND t.tag = ?)")
            params.append(tag.lower())
        if self.date_from:
            where.append("substr(COALESCE(r.occurred_at, r.created_at), 1, 10) >= ?")
            params.append(self.date_from[:10])
        if self.date_to:
            where.append("substr(COALESCE(r.occurred_at, r.created_at), 1, 10) <= ?")
            params.append(self.date_to[:10])
        if self.verified is not None:
            where.append("r.verified = ?")
            params.append(int(self.verified))
        if not self.include_demo:
            where.append("r.is_demo = 0")
        if self.status:
            where.append("lower(r.status) = lower(?)")
            params.append(self.status)
        return where, params


_TAG_LIST = "(SELECT group_concat(tag, ' ') FROM tags WHERE record_id = r.id) AS tag_list"


def search(
    conn: sqlite3.Connection,
    q: str,
    filters: Filters,
    limit: int = 50,
    offset: int = 0,
    include_documents: bool = True,
) -> dict[str, Any]:
    """Records ranked by bm25 and, separately, matching passages inside imported PDFs."""
    where, params = filters.sql()
    match = build_match_query(q)
    limit = max(1, min(limit, 500))
    if match is None:
        total = conn.execute(f"SELECT COUNT(*) FROM records r WHERE {' AND '.join(where)}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT r.*, {_TAG_LIST} FROM records r WHERE {' AND '.join(where)}"
            " ORDER BY COALESCE(r.occurred_at, r.updated_at) DESC, r.title LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        results = [{**summarize(r), "snippet": None, "score": None} for r in rows]
        return {"query": q, "mode": "browse", "total": total, "results": results}

    base = (
        "FROM records_fts JOIN records r ON r.id = records_fts.record_id"
        f" WHERE records_fts MATCH ? AND {' AND '.join(where)}"
    )
    try:
        total = conn.execute(f"SELECT COUNT(*) {base}", [match] + params).fetchone()[0]
        rows = conn.execute(
            f"SELECT r.*, {_TAG_LIST},"
            f" snippet(records_fts, -1, '{SNIPPET_OPEN}', '{SNIPPET_CLOSE}', '…', 18) AS snip,"
            " bm25(records_fts, 0.0, 10.0, 1.0, 5.0, 2.0) AS rank"
            f" {base} ORDER BY rank LIMIT ? OFFSET ?",
            [match] + params + [limit, offset],
        ).fetchall()
    except sqlite3.OperationalError as exc:  # defensive: malformed expression
        return {"query": q, "mode": "fulltext", "total": 0, "results": [], "error": str(exc)}
    results = [{**summarize(r), "snippet": r["snip"], "score": -r["rank"]} for r in rows]
    return {
        "query": q,
        "mode": "fulltext",
        "match": match,
        "total": total,
        "results": results,
        "documents": search_passages(conn, match, where, params) if include_documents else None,
    }


MAX_PASSAGE_HITS = 20_000


def search_passages(
    conn: sqlite3.Connection, match: str, where: list[str], params: list[Any], doc_limit: int = 20, per_doc: int = 3
) -> dict[str, Any]:
    """Pages of PDFs that match, grouped by PDF: the best few passages of each, PDFs ordered by
    their best passage. `where`/`params` filter the PDFs' records (alias r)."""
    try:
        hits = conn.execute(
            "SELECT p.id, p.record_id, p.page, bm25(document_pages_fts) AS rank"
            " FROM document_pages_fts JOIN document_pages p ON p.id = document_pages_fts.rowid"
            " JOIN records r ON r.id = p.record_id"
            f" WHERE document_pages_fts MATCH ? AND {' AND '.join(where)} ORDER BY rank LIMIT ?",
            [match] + params + [MAX_PASSAGE_HITS],
        ).fetchall()
    except sqlite3.OperationalError as exc:
        return {"total_pages": 0, "total_documents": 0, "capped": False, "results": [], "error": str(exc)}
    by_record: dict[str, list[sqlite3.Row]] = {}
    for hit in hits:  # best first, so each list starts with that PDF's best page
        by_record.setdefault(hit["record_id"], []).append(hit)
    shown = list(by_record.items())[:doc_limit]
    best = [h["id"] for _, record_hits in shown for h in record_hits[:per_doc]]
    snippets: dict[int, str] = {}
    if best:
        snippets = {
            row[0]: row[1]
            for row in conn.execute(
                f"SELECT rowid, snippet(document_pages_fts, 0, '{SNIPPET_OPEN}', '{SNIPPET_CLOSE}', '…', 24)"
                f" FROM document_pages_fts WHERE document_pages_fts MATCH ? AND rowid IN ({','.join('?' * len(best))})",
                [match, *best],
            )
        }
    results = []
    for record_id, record_hits in shown:
        row = conn.execute(
            f"SELECT r.*, {_TAG_LIST}, d.page_count FROM records r JOIN documents d ON d.record_id = r.id WHERE r.id = ?",
            (record_id,),
        ).fetchone()
        results.append(
            {
                **summarize(row),
                "page_count": row["page_count"],
                "pages_matched": len(record_hits),
                "passages": [
                    {"page": h["page"], "snippet": snippets.get(h["id"], ""), "score": -h["rank"]}
                    for h in record_hits[:per_doc]
                ],
            }
        )
    return {
        "total_pages": len(hits),
        "total_documents": len(by_record),
        "capped": len(hits) >= MAX_PASSAGE_HITS,
        "results": results,
    }


# ---------------------------------------------------------------- TF-IDF similarity


@dataclass
class _Index:
    signature: tuple
    idf: dict[str, float]
    vectors: dict[str, dict[str, float]]


_cache: dict[str, _Index] = {}
_lock = threading.Lock()


def _db_key(conn: sqlite3.Connection) -> str:
    return next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), "") or str(id(conn))


def _signature(conn: sqlite3.Connection) -> tuple:
    return tuple(
        conn.execute(
            "SELECT COUNT(*), MAX(updated_at), SUM(deleted_at IS NOT NULL), MAX(rowid),"
            " (SELECT MAX(extracted_at) FROM documents) FROM records"
        ).fetchone()
    )


def _weights(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(t for t in tokens if t in idf)
    vec = {t: (1 + math.log(c)) * idf[t] for t, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def _record_tokens(row: sqlite3.Row) -> list[str]:
    # Title words count three times: titles are short and usually say what a note is about.
    # For a PDF, its first two pages (title, abstract, introduction) count as well; its record
    # body alone is a short generated description.
    title = index_tokens(row["title"])
    opening = index_tokens(row["opening"]) if row["opening"] else []
    return title * 3 + index_tokens(plain_text(row["content"])) + opening + (row["tag_list"] or "").split()


def _index(conn: sqlite3.Connection) -> _Index:
    key = _db_key(conn)
    signature = _signature(conn)
    with _lock:
        cached = _cache.get(key)
        if cached and cached.signature == signature:
            return cached
    rows = conn.execute(
        f"SELECT r.id, r.title, r.content, {_TAG_LIST},"
        " (SELECT group_concat(text, ' ') FROM (SELECT text FROM document_pages p WHERE p.record_id = r.id"
        "   ORDER BY p.page LIMIT 2)) AS opening"
        " FROM records r WHERE r.deleted_at IS NULL"
    ).fetchall()
    docs = {r["id"]: _record_tokens(r) for r in rows}
    df: Counter[str] = Counter()
    for tokens in docs.values():
        df.update(set(tokens))
    n = max(len(docs), 1)
    idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
    index = _Index(signature, idf, {rid: _weights(tokens, idf) for rid, tokens in docs.items()})
    with _lock:
        _cache[key] = index
    return index


def similar(
    conn: sqlite3.Connection,
    *,
    record_id: str | None = None,
    text: str | None = None,
    filters: Filters | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    index = _index(conn)
    if record_id is not None:
        query_vec = index.vectors.get(record_id, {})
    else:
        query_vec = _weights(index_tokens(text or ""), index.idf)
    scores: list[tuple[float, str]] = []
    if query_vec:
        for rid, vec in index.vectors.items():
            if rid == record_id:
                continue
            small, large = (query_vec, vec) if len(query_vec) < len(vec) else (vec, query_vec)
            score = sum(w * large.get(t, 0.0) for t, w in small.items())
            if score > 0.02:
                scores.append((score, rid))
    scores.sort(reverse=True)
    where, params = (filters or Filters()).sql()
    results = []
    for score, rid in scores:
        row = conn.execute(
            f"SELECT r.*, {_TAG_LIST} FROM records r WHERE r.id = ? AND {' AND '.join(where)}",
            [rid] + params,
        ).fetchone()
        if row is not None:
            results.append({**summarize(row), "score": round(score, 4), "snippet": None})
        if len(results) >= limit:
            break
    return {"mode": "similar", "method": "tfidf-cosine", "total": len(results), "results": results}
