"""Candidate-contradiction detection and user-controlled resolution.

Detection is heuristic and local. It proposes pairs of statements from different records that
look like they describe the same thing but disagree:

* structured  - two facts with the same subject and predicate but different values;
* numeric     - near-identical statements that give different numbers for the same units;
* negation    - near-identical statements where exactly one is negated;
* antonym     - near-identical statements that use opposing terms (increase/decrease, ...).

Statements that mention different identifiers (dates, IDs such as DEC-003 or T01, link
targets, URLs) are treated as being about different things and are not compared. Numbers in two
records dated on different days are time-scoped (a count on Oct 21 vs Oct 31 is progress, not a
contradiction), and a line whose wording recurs in TEMPLATE_RECORDS or more records is treated as
a template field whose value is per-record. Detection
never edits or deletes a record; resolution is always an explicit user action, and even then
records are only linked ("supersedes") and annotated, never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from .db import new_id, transaction, utcnow
from .repository import InvalidInput, NotFound, create_relationship, get_record, log_activity, update_record
from .textutil import NEGATIONS, STOPWORDS, WIKILINK_RE, plain_text

RESOLUTIONS = ("prefer_a", "prefer_b", "keep_both", "not_a_conflict")

_ANTONYM_PAIRS = [
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("increasing", "decreasing"), ("higher", "lower"), ("more", "less"), ("above", "below"),
    ("before", "after"), ("accepted", "rejected"), ("approved", "rejected"), ("include", "exclude"),
    ("includes", "excludes"), ("included", "excluded"), ("enabled", "disabled"), ("true", "false"),
    ("pass", "fail"), ("passed", "failed"), ("passes", "fails"), ("valid", "invalid"),
    ("supported", "unsupported"), ("stiffer", "softer"), ("stronger", "weaker"),
    ("maximum", "minimum"), ("positive", "negative"), ("required", "optional"),
    ("retain", "remove"), ("retains", "removes"), ("retained", "removed"), ("allowed", "forbidden"),
    ("always", "never"), ("larger", "smaller"), ("greater", "smaller"), ("overestimate", "underestimate"),
]
ANTONYM_OF: dict[str, set[str]] = defaultdict(set)
for _a, _b in _ANTONYM_PAIRS:
    ANTONYM_OF[_a].add(_b)
    ANTONYM_OF[_b].add(_a)

UNITS = {
    "%", "percent", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour", "hours", "s", "sec",
    "secs", "second", "seconds", "ms", "day", "days", "week", "weeks", "year", "years", "rad", "mrad",
    "deg", "degree", "degrees", "mm", "cm", "m", "km", "in", "in.", "inch", "inches", "ft", "kip",
    "kips", "ksi", "psi", "mpa", "gpa", "kn", "n", "kg", "lb", "lbs", "g", "x", "story", "stories",
    "case", "cases", "record", "records", "shard", "shards", "design", "designs", "sample", "samples",
}

_DATE = r"\d{4}-\d{2}-\d{2}"
_CODE = r"(?:[A-Za-z]+[\w\-]*\d[\w\-]*|\d+[A-Za-z][\w\-]*)"
_IDENT_RE = re.compile(rf"\b(?:{_DATE}|{_CODE}|\d+(?:/\d+)+)\b")
_URL_RE = re.compile(r"https?://\S+")
# Things that name a source rather than state a value: inline code, file paths, and citation
# years such as "(2025)" or "(2024a)". Different ones mean different sources, not a disagreement.
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s`]+|\S+\.(?:pdf|md|py|json|csv|txt|png|jpe?g|xlsx?|docx?|zip|tcl|out)\b", re.I)
_CITATION_YEAR_RE = re.compile(r"\((?:19|20)\d{2}[a-z]?\)")
_VALUE_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?:\s*(%|[A-Za-z]+\.?))?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

MIN_CORE_WORDS = 3
MAX_DOC_FREQ = 400
TEMPLATE_RECORDS = 4
MAX_PER_PAIR = 3
BASE_SCORE = {"structured": 0.95, "numeric": 0.9, "negation": 0.85, "antonym": 0.8}


@dataclass(frozen=True)
class Statement:
    record_id: str
    text: str
    norm: str
    core: frozenset[str]
    idents: frozenset[str]
    values: tuple[tuple[float, str], ...]
    negated: bool
    antonyms: frozenset[str]


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def statements_from(content: str) -> list[str]:
    out: list[str] = []
    in_code = False
    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith("#") or line.startswith("<!--"):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(not c or re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            line = "; ".join(c for c in cells if c)
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?", "", line)
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", line):
            sentence = sentence.strip()
            if len(sentence) >= 12 and not sentence.endswith(":"):
                out.append(sentence[:500])
    return out


def analyze(record_id: str, raw: str) -> Statement | None:
    idents: set[str] = set()
    for match in WIKILINK_RE.finditer(raw):
        target = (match.group(1) or "") + (match.group(2) or "")
        idents.update(m.lower() for m in _IDENT_RE.findall(target))
    stripped = raw
    for pattern in (_CODE_SPAN_RE, _PATH_RE, _CITATION_YEAR_RE):
        idents.update(m.lower() for m in pattern.findall(stripped))
        stripped = pattern.sub(" ", stripped)
    text = plain_text(stripped)
    idents.update(u.rstrip(".,;)").lower() for u in _URL_RE.findall(text))
    text = _URL_RE.sub(" ", text)
    idents.update(m.lower() for m in _IDENT_RE.findall(text))
    text = _IDENT_RE.sub(" ", text)

    values: list[tuple[float, str]] = []

    def take_value(m: re.Match[str]) -> str:
        unit = (m.group(2) or "").lower()
        if unit in UNITS:
            values.append((float(m.group(1)), unit.rstrip(".")))
            return " "
        values.append((float(m.group(1)), ""))
        return " " + (m.group(2) or "")

    text = _VALUE_RE.sub(take_value, text)
    words = [w.lower().strip("'-") for w in _WORD_RE.findall(text)]
    negated = any(w in NEGATIONS for w in words)
    content_words = {_stem(w) for w in words if len(w) > 1 and w not in STOPWORDS and w not in NEGATIONS}
    antonyms = frozenset(w for w in words if w in ANTONYM_OF)
    core = frozenset(w for w in content_words if w not in ANTONYM_OF)
    if len(core) < MIN_CORE_WORDS:
        return None
    norm = re.sub(r"\s+", " ", plain_text(raw).lower()).strip()
    return Statement(record_id, raw, norm, core, frozenset(idents), tuple(sorted(values)), negated, antonyms)


def compare(a: Statement, b: Statement) -> tuple[str, float, str] | None:
    """(reason, score, explanation) when the two statements look contradictory."""
    if a.norm == b.norm or a.idents != b.idents:
        return None
    union = len(a.core | b.core)
    jaccard = len(a.core & b.core) / union if union else 0.0
    if jaccard < 0.6:
        return None
    if a.values and b.values and jaccard >= 0.75:
        units_a = sorted(u for _, u in a.values)
        units_b = sorted(u for _, u in b.values)
        nums_a = [v for v, _ in a.values]
        nums_b = [v for v, _ in b.values]
        if units_a == units_b and nums_a != nums_b:
            fmt = lambda vals: ", ".join(f"{v:g}{(' ' + u) if u else ''}" for v, u in vals)  # noqa: E731
            return "numeric", jaccard, f"Similar statements give different values: {fmt(a.values)} vs {fmt(b.values)}."
    if a.negated != b.negated and jaccard >= 0.75:
        which = "first" if a.negated else "second"
        return "negation", jaccard, f"Near-identical statements, but only the {which} one is negated."
    for word in a.antonyms:
        opposed = (ANTONYM_OF[word] & b.antonyms) - a.antonyms
        if opposed and word not in b.antonyms:
            return "antonym", jaccard, f"Near-identical statements use opposing terms: '{word}' vs '{sorted(opposed)[0]}'."
    return None


def _fingerprint(reason: str, a_id: str, a_text: str, b_id: str, b_text: str) -> str:
    norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()  # noqa: E731
    (x_id, x_text), (y_id, y_text) = sorted([(a_id, norm(a_text)), (b_id, norm(b_text))])
    return hashlib.sha1(f"{reason}|{x_id}|{x_text}|{y_id}|{y_text}".encode("utf-8")).hexdigest()


def _superseded_pairs(conn: sqlite3.Connection) -> set[frozenset[str]]:
    return {
        frozenset((r[0], r[1]))
        for r in conn.execute("SELECT source_id, target_id FROM relationships WHERE rel_type = 'supersedes'")
    }


def detect(
    conn: sqlite3.Connection, record_ids: Iterable[str] | None = None, max_new: int = 200
) -> dict[str, Any]:
    """Scan for new candidate conflicts and store them as 'open'. Never modifies records."""
    focus = set(record_ids) if record_ids is not None else None
    # A PDF's record body is generated ("PDF · 23 pages …"), not a statement to compare.
    rows = conn.execute(
        "SELECT id, title, content, is_demo, subject, predicate, value, occurred_at"
        " FROM records WHERE deleted_at IS NULL AND id NOT IN (SELECT record_id FROM documents)"
    ).fetchall()
    demo = {r["id"]: r["is_demo"] for r in rows}
    day = {r["id"]: (r["occurred_at"] or "")[:10] for r in rows}
    skip_pairs = _superseded_pairs(conn)
    known = {r[0] for r in conn.execute("SELECT fingerprint FROM conflicts")}
    candidates: list[dict[str, Any]] = []

    def eligible(a_id: str, b_id: str) -> bool:
        if a_id == b_id or demo[a_id] != demo[b_id] or frozenset((a_id, b_id)) in skip_pairs:
            return False
        return focus is None or a_id in focus or b_id in focus

    # Structured facts: same subject + predicate, different value.
    groups: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        if r["subject"] and r["predicate"] and r["value"]:
            groups[(r["subject"].strip().lower(), r["predicate"].strip().lower())].append(r)
    for (subject, predicate), members in groups.items():
        for a, b in combinations(members, 2):
            if not eligible(a["id"], b["id"]) or a["value"].strip().lower() == b["value"].strip().lower():
                continue
            ex_a = f"{a['subject']} · {a['predicate']} = {a['value']}"
            ex_b = f"{b['subject']} · {b['predicate']} = {b['value']}"
            candidates.append(
                {
                    "a": a["id"], "b": b["id"], "excerpt_a": ex_a, "excerpt_b": ex_b, "reason": "structured",
                    "score": BASE_SCORE["structured"],
                    "explanation": f"Both facts describe '{a['subject']} · {a['predicate']}' with different values.",
                }
            )

    # Statement-level comparison with an inverted index for blocking.
    statements: list[Statement] = []
    for r in rows:
        for raw in statements_from(r["content"]):
            st = analyze(r["id"], raw)
            if st is not None:
                statements.append(st)
    df: Counter[str] = Counter()
    for st in statements:
        df.update(st.core)
    shape_records: dict[tuple[frozenset[str], frozenset[str]], set[str]] = defaultdict(set)
    for st in statements:
        shape_records[(st.core, st.idents)].add(st.record_id)
    templated = {shape for shape, ids in shape_records.items() if len(ids) >= TEMPLATE_RECORDS}
    postings: dict[str, list[int]] = defaultdict(list)
    for i, st in enumerate(statements):
        for w in st.core:
            if df[w] <= MAX_DOC_FREQ:
                postings[w].append(i)
    for i, a in enumerate(statements):
        overlap: Counter[int] = Counter()
        for w in a.core:
            if df[w] <= MAX_DOC_FREQ:
                overlap.update(j for j in postings[w] if j > i)
        for j, shared in overlap.items():
            b = statements[j]
            # Jaccard >= 0.6 needs at least this many shared words.
            if shared < 0.6 * max(len(a.core), len(b.core)) or not eligible(a.record_id, b.record_id):
                continue
            if (a.core, a.idents) in templated or (b.core, b.idents) in templated:
                continue
            verdict = compare(a, b)
            if verdict:
                reason, jaccard, explanation = verdict
                if reason == "numeric" and day[a.record_id] and day[b.record_id] and day[a.record_id] != day[b.record_id]:
                    continue
                candidates.append(
                    {
                        "a": a.record_id, "b": b.record_id, "excerpt_a": a.text, "excerpt_b": b.text,
                        "reason": reason, "score": round(BASE_SCORE[reason] * jaccard, 4), "explanation": explanation,
                    }
                )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    per_pair: Counter[frozenset[str]] = Counter()
    created: list[str] = []
    with transaction(conn):
        for c in candidates:
            if len(created) >= max_new:
                break
            pair = frozenset((c["a"], c["b"]))
            fp = _fingerprint(c["reason"], c["a"], c["excerpt_a"], c["b"], c["excerpt_b"])
            if fp in known or per_pair[pair] >= MAX_PER_PAIR:
                continue
            a_id, b_id, ex_a, ex_b = c["a"], c["b"], c["excerpt_a"], c["excerpt_b"]
            if a_id > b_id:
                a_id, b_id, ex_a, ex_b = b_id, a_id, ex_b, ex_a
            conflict_id = new_id()
            conn.execute(
                "INSERT INTO conflicts (id, record_a_id, record_b_id, excerpt_a, excerpt_b, reason, explanation,"
                " score, fingerprint, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (conflict_id, a_id, b_id, ex_a, ex_b, c["reason"], c["explanation"], c["score"], fp, utcnow()),
            )
            known.add(fp)
            per_pair[pair] += 1
            created.append(conflict_id)
        if created:
            log_activity(conn, "conflicts-detected", None, None, f"{len(created)} new candidate(s)")
    return {
        "scanned_records": len(rows) if focus is None else len(focus),
        "statements": len(statements),
        "candidates": len(candidates),
        "new_conflicts": created,
    }


# ---------------------------------------------------------------- review


def _conflict_row(conn: sqlite3.Connection, conflict_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM conflicts WHERE id = ?", (conflict_id,)).fetchone()
    if row is None:
        raise NotFound(f"conflict {conflict_id} not found")
    return row


def list_conflicts(conn: sqlite3.Connection, status: str | None = None, include_demo: bool = True) -> list[dict[str, Any]]:
    where, params = [], []
    if status:
        where.append("c.status = ?")
        params.append(status)
    if not include_demo:
        where.append("a.is_demo = 0 AND b.is_demo = 0")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        "SELECT c.*, a.title AS a_title, a.category AS a_category, a.deleted_at AS a_deleted, a.is_demo AS a_demo,"
        "       b.title AS b_title, b.category AS b_category, b.deleted_at AS b_deleted"
        " FROM conflicts c JOIN records a ON a.id = c.record_a_id JOIN records b ON b.id = c.record_b_id"
        f" {clause} ORDER BY CASE c.status WHEN 'open' THEN 0 ELSE 1 END, c.score DESC, c.detected_at DESC",
        params,
    ).fetchall()
    return [
        {
            "id": r["id"],
            "status": r["status"],
            "reason": r["reason"],
            "explanation": r["explanation"],
            "score": r["score"],
            "excerpt_a": r["excerpt_a"],
            "excerpt_b": r["excerpt_b"],
            "resolution": r["resolution"],
            "resolution_note": r["resolution_note"],
            "detected_at": r["detected_at"],
            "resolved_at": r["resolved_at"],
            "is_demo": bool(r["a_demo"]),
            "a": {"id": r["record_a_id"], "title": r["a_title"], "category": r["a_category"], "deleted": r["a_deleted"] is not None},
            "b": {"id": r["record_b_id"], "title": r["b_title"], "category": r["b_category"], "deleted": r["b_deleted"] is not None},
        }
        for r in rows
    ]


def get_conflict(conn: sqlite3.Connection, conflict_id: str) -> dict[str, Any]:
    row = _conflict_row(conn, conflict_id)
    out = {key: row[key] for key in row.keys()}
    out["record_a"] = get_record(conn, row["record_a_id"])
    out["record_b"] = get_record(conn, row["record_b_id"])
    return out


def flag_conflict(
    conn: sqlite3.Connection, a_id: str, b_id: str, excerpt_a: str, excerpt_b: str, explanation: str
) -> str:
    if a_id == b_id:
        raise InvalidInput("a conflict needs two different records")
    for rid in (a_id, b_id):
        if conn.execute("SELECT 1 FROM records WHERE id = ? AND deleted_at IS NULL", (rid,)).fetchone() is None:
            raise InvalidInput(f"record {rid} does not exist or is in the trash")
    conflict_id = new_id()
    with transaction(conn):
        conn.execute(
            "INSERT INTO conflicts (id, record_a_id, record_b_id, excerpt_a, excerpt_b, reason, explanation, score,"
            " fingerprint, detected_at) VALUES (?, ?, ?, ?, ?, 'manual', ?, 1.0, ?, ?)",
            (conflict_id, a_id, b_id, excerpt_a.strip() or "(whole record)", excerpt_b.strip() or "(whole record)",
             explanation.strip() or "Flagged manually.", f"manual-{conflict_id}", utcnow()),
        )
        title = conn.execute("SELECT title FROM records WHERE id = ?", (a_id,)).fetchone()[0]
        log_activity(conn, "conflict-flagged", a_id, title, "manual")
    return conflict_id


def resolve_conflict(conn: sqlite3.Connection, conflict_id: str, resolution: str, note: str | None = None) -> None:
    """Record the user's decision. Preferring one side links it as superseding the other and
    annotates the other's metadata; neither record's content is changed or deleted."""
    if resolution not in RESOLUTIONS:
        raise InvalidInput(f"resolution must be one of {', '.join(RESOLUTIONS)}")
    with transaction(conn):
        c = _conflict_row(conn, conflict_id)
        if c["status"] != "open":
            raise InvalidInput("conflict is already closed; reopen it first")
        rel_id = None
        keep = None
        if resolution in ("prefer_a", "prefer_b"):
            keep, other = (
                (c["record_a_id"], c["record_b_id"]) if resolution == "prefer_a" else (c["record_b_id"], c["record_a_id"])
            )
            already = conn.execute(
                "SELECT 1 FROM relationships WHERE source_id = ? AND target_id = ? AND rel_type = 'supersedes'",
                (keep, other),
            ).fetchone()
            new_rel = create_relationship(
                conn, keep, other, "supersedes", note=f"Conflict review: {note or resolution}", origin="conflict-resolution"
            )
            rel_id = None if already else new_rel
            meta = json.loads(conn.execute("SELECT metadata FROM records WHERE id = ?", (other,)).fetchone()[0])
            meta["superseded_by"] = keep
            update_record(conn, other, {"metadata": meta}, reason=f"conflict {conflict_id} resolved in favour of {keep}")
        conn.execute(
            "UPDATE conflicts SET status = ?, resolution = ?, resolution_note = ?, resolution_relationship_id = ?,"
            " resolved_at = ? WHERE id = ?",
            ("dismissed" if resolution == "not_a_conflict" else "resolved", resolution, note, rel_id, utcnow(), conflict_id),
        )
        title = conn.execute("SELECT title FROM records WHERE id = ?", (keep or c["record_a_id"],)).fetchone()[0]
        log_activity(conn, "conflict-resolved", keep or c["record_a_id"], title, resolution)


def reopen_conflict(conn: sqlite3.Connection, conflict_id: str) -> None:
    """Undo a resolution: removes the link and annotation that resolution added."""
    with transaction(conn):
        c = _conflict_row(conn, conflict_id)
        if c["status"] == "open":
            return
        if c["resolution"] in ("prefer_a", "prefer_b"):
            keep, other = (
                (c["record_a_id"], c["record_b_id"]) if c["resolution"] == "prefer_a" else (c["record_b_id"], c["record_a_id"])
            )
            if c["resolution_relationship_id"]:
                conn.execute("DELETE FROM relationships WHERE id = ?", (c["resolution_relationship_id"],))
            row = conn.execute("SELECT metadata, deleted_at FROM records WHERE id = ?", (other,)).fetchone()
            meta = json.loads(row["metadata"])
            if meta.get("superseded_by") == keep and row["deleted_at"] is None:
                meta.pop("superseded_by")
                update_record(conn, other, {"metadata": meta}, reason=f"conflict {conflict_id} reopened")
        conn.execute(
            "UPDATE conflicts SET status = 'open', resolution = NULL, resolution_note = NULL,"
            " resolution_relationship_id = NULL, resolved_at = NULL WHERE id = ?",
            (conflict_id,),
        )
        log_activity(conn, "conflict-reopened", c["record_a_id"], None, conflict_id)
