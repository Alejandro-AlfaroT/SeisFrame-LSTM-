"""Records, tags, relationships, version history and the activity log."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Iterable

from .db import later_than, new_id, transaction, utcnow
from .textutil import excerpt, sanitize_text

CATEGORIES = ("facts", "sessions", "inbox", "projects")

# Organizational metaphor only: the brain region of the visualization that shows each
# category. Conflicts are pairs of records, not records, so they have a region but no category.
REGIONS = {
    "facts": "hippocampus",
    "sessions": "occipital",
    "inbox": "temporal",
    "conflicts": "frontal",
    "projects": "prefrontal",
}

REL_TYPES = (
    "relates_to",
    "references",
    "supports",
    "contradicts",
    "depends_on",
    "part_of",
    "supersedes",
    "derived_from",
    "decided_in",
    "follows",
)

ACTIVE_PROJECT_STATUSES = ("active", "in-progress", "ongoing")

RECORD_FIELDS = (
    "title",
    "content",
    "category",
    "kind",
    "status",
    "project_id",
    "source_ref",
    "confidence",
    "verified",
    "subject",
    "predicate",
    "value",
    "occurred_at",
    "metadata",
)

KIND_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?$"
)


class NotFound(Exception):
    pass


class InvalidInput(ValueError):
    pass


class StaleWrite(Exception):
    """The record changed since the client loaded it; refusing to overwrite."""


# ---------------------------------------------------------------- validation


def normalize_tags(tags: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in tags:
        tag = str(raw).strip().lstrip("#").strip().lower()
        tag = re.sub(r"\s+", "-", tag)
        tag = re.sub(r"[^\w\-/.]", "", tag)[:64]
        if tag and tag not in out:
            out.append(tag)
    return out[:50]


def _clean(conn: sqlite3.Connection, data: dict[str, Any], record_id: str | None) -> dict[str, Any]:
    unknown = set(data) - set(RECORD_FIELDS) - {"tags"}
    if unknown:
        raise InvalidInput(f"Unknown fields: {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key == "tags":
            continue
        if key == "title":
            value = sanitize_text(str(value or "")).strip()
            if not value:
                raise InvalidInput("title must not be empty")
            if len(value) > 500:
                raise InvalidInput("title is longer than 500 characters")
        elif key == "content":
            value = sanitize_text(str(value or ""))
        elif key == "category":
            if value not in CATEGORIES:
                raise InvalidInput(f"category must be one of {', '.join(CATEGORIES)}")
        elif key == "kind":
            value = str(value or "note").strip().lower()
            if not KIND_RE.match(value):
                raise InvalidInput("kind must be a short lowercase slug, e.g. 'fact' or 'decision'")
        elif key in ("status", "source_ref", "subject", "predicate", "value"):
            value = sanitize_text(str(value)).strip()[:2000] if value is not None else None
            value = value or None
        elif key == "confidence":
            if value is not None:
                value = float(value)
                if not 0.0 <= value <= 1.0:
                    raise InvalidInput("confidence must be between 0 and 1")
        elif key == "verified":
            value = bool(value)
        elif key == "occurred_at":
            value = str(value).strip() if value else None
            if value and not DATE_RE.match(value):
                raise InvalidInput("occurred_at must be an ISO date or datetime")
        elif key == "project_id":
            value = value or None
            if value is not None:
                if value == record_id:
                    raise InvalidInput("a project cannot belong to itself")
                row = conn.execute(
                    "SELECT kind, deleted_at FROM records WHERE id = ?", (value,)
                ).fetchone()
                if row is None or row["deleted_at"] is not None or row["kind"] != "project":
                    raise InvalidInput("project_id must reference an existing project record")
        elif key == "metadata":
            value = value or {}
            if not isinstance(value, dict):
                raise InvalidInput("metadata must be an object")
        out[key] = value
    return out


def _db_value(key: str, value: Any) -> Any:
    if key == "metadata":
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if key == "verified":
        return int(bool(value))
    return value


# ---------------------------------------------------------------- reading


def _row(conn: sqlite3.Connection, record_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        raise NotFound(f"record {record_id} not found")
    return row


def record_tags(conn: sqlite3.Connection, record_id: str) -> list[str]:
    return [r[0] for r in conn.execute("SELECT tag FROM tags WHERE record_id = ? ORDER BY tag", (record_id,))]


def summarize(row: sqlite3.Row | dict[str, Any], tags: list[str] | None = None) -> dict[str, Any]:
    """Compact form used by lists, search results and the brain view."""
    keys = row.keys()
    tag_list = tags
    if tag_list is None:
        tag_list = (row["tag_list"] or "").split() if "tag_list" in keys else []
    meta = json.loads(row["metadata"]) if "metadata" in keys else {}
    return {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "region": REGIONS[row["category"]],
        "kind": row["kind"],
        "status": row["status"],
        "project_id": row["project_id"],
        "occurred_at": row["occurred_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "verified": bool(row["verified"]),
        "is_demo": bool(row["is_demo"]),
        "deleted": row["deleted_at"] is not None,
        "tags": tag_list,
        "excerpt": excerpt(row["content"]) if "content" in keys else "",
        "suggested_category": meta.get("suggested_category"),
        "superseded_by": meta.get("superseded_by"),
    }


def get_record(conn: sqlite3.Connection, record_id: str) -> dict[str, Any]:
    row = _row(conn, record_id)
    record = {key: row[key] for key in row.keys()}
    record["metadata"] = json.loads(record["metadata"])
    record["verified"] = bool(record["verified"])
    record["is_demo"] = bool(record["is_demo"])
    record["deleted"] = record["deleted_at"] is not None
    record["region"] = REGIONS[record["category"]]
    record["tags"] = record_tags(conn, record_id)
    project = None
    if record["project_id"]:
        p = conn.execute("SELECT id, title FROM records WHERE id = ?", (record["project_id"],)).fetchone()
        project = {"id": p["id"], "title": p["title"]} if p else None
    record["project"] = project
    record["relationships"] = relationships_for(conn, record_id)
    record["version_count"] = conn.execute(
        "SELECT COUNT(*) FROM record_versions WHERE record_id = ?", (record_id,)
    ).fetchone()[0]
    record["conflicts"] = [
        dict(r)
        for r in conn.execute(
            "SELECT id, status, reason, record_a_id, record_b_id FROM conflicts"
            " WHERE record_a_id = ? OR record_b_id = ? ORDER BY detected_at DESC",
            (record_id, record_id),
        )
    ]
    return record


def find_by_source_ref(conn: sqlite3.Connection, source_ref: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM records WHERE source_ref = ? ORDER BY created_at LIMIT 1", (source_ref,)
    ).fetchone()


# ---------------------------------------------------------------- writing


def _set_tags(conn: sqlite3.Connection, record_id: str, tags: list[str]) -> None:
    conn.execute("DELETE FROM tags WHERE record_id = ?", (record_id,))
    conn.executemany("INSERT INTO tags (record_id, tag) VALUES (?, ?)", [(record_id, t) for t in tags])


def reindex(conn: sqlite3.Connection, record_id: str) -> None:
    conn.execute("DELETE FROM records_fts WHERE record_id = ?", (record_id,))
    conn.execute(
        "INSERT INTO records_fts (record_id, title, content, tags, source_ref)"
        " SELECT r.id, r.title, r.content,"
        "        COALESCE((SELECT group_concat(tag, ' ') FROM tags WHERE record_id = r.id), ''),"
        "        COALESCE(r.source_ref, '')"
        " FROM records r WHERE r.id = ?",
        (record_id,),
    )


def log_activity(
    conn: sqlite3.Connection, action: str, record_id: str | None, title: str | None, detail: str | None = None
) -> None:
    conn.execute(
        "INSERT INTO activity (at, action, record_id, title, detail) VALUES (?, ?, ?, ?, ?)",
        (utcnow(), action, record_id, title, detail),
    )


def create_record(
    conn: sqlite3.Connection,
    data: dict[str, Any],
    *,
    is_demo: bool = False,
    import_batch_id: str | None = None,
    source_hash: str | None = None,
    log: bool = True,
) -> str:
    if "title" not in data or "category" not in data:
        raise InvalidInput("title and category are required")
    fields = _clean(conn, data, None)
    tags = normalize_tags(data.get("tags") or [])
    record_id = new_id()
    now = utcnow()
    values = {
        "content": "",
        "kind": "note",
        "status": None,
        "project_id": None,
        "source_ref": None,
        "confidence": None,
        "verified": False,
        "subject": None,
        "predicate": None,
        "value": None,
        "occurred_at": None,
        "metadata": {},
        **fields,
    }
    with transaction(conn):
        conn.execute(
            "INSERT INTO records (id, title, content, category, kind, status, project_id, source_ref,"
            " source_hash, confidence, verified, verified_at, subject, predicate, value, occurred_at,"
            " metadata, is_demo, import_batch_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record_id,
                values["title"],
                values["content"],
                values["category"],
                values["kind"],
                values["status"],
                values["project_id"],
                values["source_ref"],
                source_hash,
                values["confidence"],
                int(values["verified"]),
                now if values["verified"] else None,
                values["subject"],
                values["predicate"],
                values["value"],
                values["occurred_at"],
                _db_value("metadata", values["metadata"]),
                int(is_demo),
                import_batch_id,
                now,
                now,
            ),
        )
        _set_tags(conn, record_id, tags)
        reindex(conn, record_id)
        if log:
            log_activity(conn, "created", record_id, values["title"], f"{values['category']}/{values['kind']}")
    return record_id


def _snapshot(conn: sqlite3.Connection, row: sqlite3.Row, reason: str) -> None:
    snapshot = {key: row[key] for key in row.keys()}
    snapshot["tags"] = record_tags(conn, row["id"])
    conn.execute(
        "INSERT INTO record_versions (record_id, snapshot, reason, created_at) VALUES (?, ?, ?, ?)",
        (row["id"], json.dumps(snapshot, ensure_ascii=False), reason, utcnow()),
    )


def update_record(
    conn: sqlite3.Connection,
    record_id: str,
    patch: dict[str, Any],
    *,
    reason: str = "edit",
    expected_updated_at: str | None = None,
    source_hash: str | None = None,
    log: bool = True,
) -> bool:
    """Apply a partial update. The previous state is kept in record_versions.

    Returns False when nothing changed.
    """
    with transaction(conn):
        row = _row(conn, record_id)
        if row["deleted_at"] is not None:
            raise InvalidInput("record is in the trash; restore it before editing")
        if expected_updated_at is not None and row["updated_at"] != expected_updated_at:
            raise StaleWrite("record was changed elsewhere; reload before saving")
        fields = _clean(conn, patch, record_id)
        changed = {k: v for k, v in fields.items() if _db_value(k, v) != row[k]}
        new_tags = normalize_tags(patch["tags"]) if patch.get("tags") is not None else None
        tags_changed = new_tags is not None and new_tags != record_tags(conn, record_id)
        hash_changed = source_hash is not None and source_hash != row["source_hash"]
        if not changed and not tags_changed and not hash_changed:
            return False
        _snapshot(conn, row, reason)
        now = later_than(row["updated_at"])
        if "verified" in changed:
            changed["verified_at"] = now if changed["verified"] else None
        if hash_changed:
            changed["source_hash"] = source_hash
        changed["updated_at"] = now
        assignments = ", ".join(f"{key} = ?" for key in changed)
        conn.execute(
            f"UPDATE records SET {assignments} WHERE id = ?",
            [_db_value(k, v) for k, v in changed.items()] + [record_id],
        )
        if tags_changed:
            _set_tags(conn, record_id, new_tags or [])
        reindex(conn, record_id)
        if log:
            fields_changed = sorted(set(changed) - {"updated_at", "verified_at", "source_hash"})
            if tags_changed:
                fields_changed.append("tags")
            log_activity(conn, "updated", record_id, fields.get("title", row["title"]), ", ".join(fields_changed))
    return True


def trash_record(conn: sqlite3.Connection, record_id: str) -> None:
    with transaction(conn):
        row = _row(conn, record_id)
        if row["deleted_at"] is None:
            conn.execute("UPDATE records SET deleted_at = ? WHERE id = ?", (utcnow(), record_id))
            log_activity(conn, "trashed", record_id, row["title"])


def restore_record(conn: sqlite3.Connection, record_id: str) -> None:
    with transaction(conn):
        row = _row(conn, record_id)
        if row["deleted_at"] is not None:
            conn.execute("UPDATE records SET deleted_at = NULL WHERE id = ?", (record_id,))
            log_activity(conn, "restored", record_id, row["title"])


def purge_record(conn: sqlite3.Connection, record_id: str) -> None:
    """Permanent removal. Only records already in the trash can be purged."""
    with transaction(conn):
        row = _row(conn, record_id)
        if row["deleted_at"] is None:
            raise InvalidInput("only records in the trash can be permanently deleted")
        conn.execute("DELETE FROM records_fts WHERE record_id = ?", (record_id,))
        conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        log_activity(conn, "purged", record_id, row["title"])


def record_versions(conn: sqlite3.Connection, record_id: str) -> list[dict[str, Any]]:
    _row(conn, record_id)
    return [
        {"id": r["id"], "reason": r["reason"], "created_at": r["created_at"], "snapshot": json.loads(r["snapshot"])}
        for r in conn.execute(
            "SELECT * FROM record_versions WHERE record_id = ? ORDER BY id DESC", (record_id,)
        )
    ]


def revert_record(conn: sqlite3.Connection, record_id: str, version_id: int) -> None:
    """Restore content fields from a snapshot. The current state is itself snapshotted first."""
    row = conn.execute(
        "SELECT snapshot FROM record_versions WHERE id = ? AND record_id = ?", (version_id, record_id)
    ).fetchone()
    if row is None:
        raise NotFound(f"version {version_id} not found for record {record_id}")
    snap = json.loads(row["snapshot"])
    patch = {key: snap[key] for key in RECORD_FIELDS if key in snap}
    patch["metadata"] = json.loads(patch["metadata"]) if isinstance(patch.get("metadata"), str) else patch.get("metadata")
    patch["tags"] = snap.get("tags", [])
    if patch.get("project_id"):
        project = conn.execute(
            "SELECT 1 FROM records WHERE id = ? AND kind = 'project' AND deleted_at IS NULL", (patch["project_id"],)
        ).fetchone()
        if project is None:
            patch["project_id"] = None
    update_record(conn, record_id, patch, reason=f"revert to version {version_id}")


# ---------------------------------------------------------------- relationships


def _relationship(conn: sqlite3.Connection, rel_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM relationships WHERE id = ?", (rel_id,)).fetchone()
    if row is None:
        raise NotFound(f"relationship {rel_id} not found")
    return row


def create_relationship(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    rel_type: str,
    note: str | None = None,
    origin: str = "user",
    log: bool = True,
) -> str:
    if rel_type not in REL_TYPES:
        raise InvalidInput(f"rel_type must be one of {', '.join(REL_TYPES)}")
    if source_id == target_id:
        raise InvalidInput("a record cannot be related to itself")
    with transaction(conn):
        source = _row(conn, source_id)
        target = _row(conn, target_id)
        if source["deleted_at"] or target["deleted_at"]:
            raise InvalidInput("cannot link records in the trash")
        existing = conn.execute(
            "SELECT id FROM relationships WHERE source_id = ? AND target_id = ? AND rel_type = ?",
            (source_id, target_id, rel_type),
        ).fetchone()
        if existing:
            return existing["id"]
        rel_id = new_id()
        now = utcnow()
        conn.execute(
            "INSERT INTO relationships (id, source_id, target_id, rel_type, note, origin, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rel_id, source_id, target_id, rel_type, note, origin, now, now),
        )
        if log:
            log_activity(conn, "linked", source_id, source["title"], f"{rel_type} → {target['title']}")
    return rel_id


def update_relationship(
    conn: sqlite3.Connection, rel_id: str, rel_type: str | None = None, note: str | None = None
) -> None:
    with transaction(conn):
        row = _relationship(conn, rel_id)
        new_type = rel_type or row["rel_type"]
        if new_type not in REL_TYPES:
            raise InvalidInput(f"rel_type must be one of {', '.join(REL_TYPES)}")
        try:
            conn.execute(
                "UPDATE relationships SET rel_type = ?, note = ?, updated_at = ? WHERE id = ?",
                (new_type, note if note is not None else row["note"], utcnow(), rel_id),
            )
        except sqlite3.IntegrityError as exc:
            raise InvalidInput("an identical relationship already exists") from exc
        title = conn.execute("SELECT title FROM records WHERE id = ?", (row["source_id"],)).fetchone()[0]
        log_activity(conn, "relinked", row["source_id"], title, f"{row['rel_type']} → {new_type}")


def delete_relationship(conn: sqlite3.Connection, rel_id: str) -> None:
    with transaction(conn):
        row = _relationship(conn, rel_id)
        conn.execute("DELETE FROM relationships WHERE id = ?", (rel_id,))
        title = conn.execute("SELECT title FROM records WHERE id = ?", (row["source_id"],)).fetchone()[0]
        log_activity(conn, "unlinked", row["source_id"], title, row["rel_type"])


def relationships_for(conn: sqlite3.Connection, record_id: str) -> dict[str, list[dict[str, Any]]]:
    def fetch(direction: str) -> list[dict[str, Any]]:
        mine, other = ("source_id", "target_id") if direction == "outgoing" else ("target_id", "source_id")
        rows = conn.execute(
            f"SELECT rel.*, o.title AS other_title, o.category AS other_category, o.kind AS other_kind,"
            f"       o.deleted_at AS other_deleted"
            f" FROM relationships rel JOIN records o ON o.id = rel.{other}"
            f" WHERE rel.{mine} = ? ORDER BY rel.rel_type, o.title",
            (record_id,),
        )
        return [
            {
                "id": r["id"],
                "rel_type": r["rel_type"],
                "note": r["note"],
                "origin": r["origin"],
                "other": {
                    "id": r[other],
                    "title": r["other_title"],
                    "category": r["other_category"],
                    "kind": r["other_kind"],
                    "deleted": r["other_deleted"] is not None,
                },
            }
            for r in rows
        ]

    return {"outgoing": fetch("outgoing"), "incoming": fetch("incoming")}


# ---------------------------------------------------------------- aggregates


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    by_category = {c: 0 for c in CATEGORIES}
    for row in conn.execute(
        "SELECT category, COUNT(*) FROM records WHERE deleted_at IS NULL GROUP BY category"
    ):
        by_category[row[0]] = row[1]
    one = lambda sql, *params: conn.execute(sql, params).fetchone()[0]  # noqa: E731
    placeholders = ",".join("?" * len(ACTIVE_PROJECT_STATUSES))
    return {
        "total_records": sum(by_category.values()),
        "by_category": by_category,
        "verified_facts": one(
            "SELECT COUNT(*) FROM records WHERE deleted_at IS NULL AND category = 'facts' AND verified = 1"
        ),
        "unverified_facts": one(
            "SELECT COUNT(*) FROM records WHERE deleted_at IS NULL AND category = 'facts' AND verified = 0"
        ),
        "active_projects": one(
            "SELECT COUNT(*) FROM records WHERE deleted_at IS NULL AND kind = 'project'"
            f" AND (status IS NULL OR lower(status) IN ({placeholders}))",
            *ACTIVE_PROJECT_STATUSES,
        ),
        "sessions": by_category["sessions"],
        "inbox": by_category["inbox"],
        "open_conflicts": one(
            "SELECT COUNT(*) FROM conflicts c JOIN records a ON a.id = c.record_a_id"
            " JOIN records b ON b.id = c.record_b_id"
            " WHERE c.status = 'open' AND a.deleted_at IS NULL AND b.deleted_at IS NULL"
        ),
        "relationships": one("SELECT COUNT(*) FROM relationships"),
        "trash": one("SELECT COUNT(*) FROM records WHERE deleted_at IS NOT NULL"),
        "demo_records": one("SELECT COUNT(*) FROM records WHERE deleted_at IS NULL AND is_demo = 1"),
    }


def recent_activity(conn: sqlite3.Connection, limit: int = 30) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))]


def list_tags(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {"tag": r[0], "count": r[1]}
        for r in conn.execute(
            "SELECT t.tag, COUNT(*) FROM tags t JOIN records r ON r.id = t.record_id"
            " WHERE r.deleted_at IS NULL GROUP BY t.tag ORDER BY COUNT(*) DESC, t.tag"
        )
    ]


def list_projects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    projects = []
    for p in conn.execute(
        "SELECT * FROM records WHERE kind = 'project' AND deleted_at IS NULL ORDER BY title"
    ):
        kinds = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT kind, COUNT(*) FROM records WHERE project_id = ? AND deleted_at IS NULL GROUP BY kind",
                (p["id"],),
            )
        }
        milestones = [
            {"id": m["id"], "title": m["title"], "status": m["status"], "occurred_at": m["occurred_at"]}
            for m in conn.execute(
                "SELECT id, title, status, occurred_at FROM records WHERE project_id = ? AND kind IN"
                " ('milestone', 'checkpoint') AND deleted_at IS NULL ORDER BY COALESCE(occurred_at, created_at)",
                (p["id"],),
            )
        ]
        meta = json.loads(p["metadata"])
        projects.append(
            {
                **summarize(p),
                "objectives": meta.get("objectives", []),
                "record_counts": kinds,
                "milestones": milestones,
            }
        )
    return projects


def list_trash(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {**summarize(r), "deleted_at": r["deleted_at"]}
        for r in conn.execute(
            "SELECT r.*, (SELECT group_concat(tag, ' ') FROM tags WHERE record_id = r.id) AS tag_list"
            " FROM records r WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC"
        )
    ]


def resolve_reference(conn: sqlite3.Connection, ref: str) -> dict[str, Any] | None:
    """Find the record a wikilink-style reference points at (source path, path stem, or title)."""
    ref = ref.split("#", 1)[0].strip().replace("\\", "/")
    if not ref:
        return None
    stem = ref[:-3] if ref.lower().endswith(".md") else ref
    candidates = conn.execute(
        "SELECT r.*, NULL AS tag_list FROM records r WHERE deleted_at IS NULL AND ("
        " source_ref = ? OR source_ref LIKE ? ESCAPE '\\' OR lower(title) = lower(?))"
        " ORDER BY CASE WHEN source_ref LIKE ? ESCAPE '\\' THEN 0 ELSE 1 END, created_at",
        (stem + ".md", "%/" + _like_escape(stem) + ".md", stem.rsplit("/", 1)[-1], "%/" + _like_escape(stem) + ".md"),
    ).fetchall()
    return summarize(candidates[0]) if candidates else None


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
