"""Explicit Markdown import (single files, uploads, or an Obsidian-style folder).

Imported documents are untrusted:
* only .md/.markdown files are read, each at most MAX_FILE_BYTES; symlinks and paths that
  resolve outside the chosen folder are skipped; hidden folders are never entered;
* YAML frontmatter is parsed with yaml.safe_load and reduced to bounded, JSON-safe values;
* content is stored as text and never executed; the UI renders it without raw HTML;
* nothing is marked verified, and a dry run previews every action without writing.
Re-importing the same source updates unchanged-in-CORTEX records (the previous state is kept
in version history) and skips records that were edited inside CORTEX since their import.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from . import conflicts
from .db import new_id, transaction, utcnow
from .repository import (
    DATE_RE,
    InvalidInput,
    create_record,
    create_relationship,
    find_by_source_ref,
    log_activity,
    normalize_tags,
    record_tags,
    update_record,
)
from .textutil import WIKILINK_RE, sanitize_text

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_FILES = 5000
MARKDOWN_SUFFIXES = (".md", ".markdown")
DEFAULT_EXCLUDED_DIRS = ("templates",)
ROUTES = ("inbox", "by_type")

TYPE_ROUTES: dict[str, tuple[str, str]] = {
    "decision": ("projects", "decision"),
    "checkpoint": ("projects", "milestone"),
    "milestone": ("projects", "milestone"),
    "experiment": ("projects", "experiment"),
    "agent-task": ("projects", "task"),
    "agent-return": ("projects", "task"),
    "task": ("projects", "task"),
    "roadmap": ("projects", "plan"),
    "dashboard": ("projects", "plan"),
    "research-scope": ("projects", "plan"),
    "research-protocol": ("projects", "protocol"),
    "project": ("projects", "project"),
    "daily": ("sessions", "session"),
    "research-session": ("sessions", "session"),
    "session": ("sessions", "session"),
    "review": ("sessions", "review"),
    "literature": ("facts", "literature"),
    "concept": ("facts", "concept"),
    "fact": ("facts", "fact"),
}
FOLDER_ROUTES: dict[str, tuple[str, str]] = {
    "planning": ("projects", "plan"),
    "research": ("facts", "note"),
    "literature": ("facts", "literature"),
    "experiments": ("projects", "experiment"),
    "agents": ("projects", "task"),
    "decisions": ("projects", "decision"),
    "daily": ("sessions", "session"),
}


@dataclass
class SourceFile:
    path: str
    content: str


@dataclass
class ImportOptions:
    route: str = "inbox"
    project_id: str | None = None
    source_label: str = "import"
    dry_run: bool = False
    update_edited: bool = False
    extra_tags: list[str] = field(default_factory=list)
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDED_DIRS
    root: str | None = None


class _DryRun(Exception):
    pass


# ---------------------------------------------------------------- reading sources


def _decode(data: bytes) -> tuple[str, list[str]]:
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), ["file is not valid UTF-8; undecodable bytes replaced"]


def _excluded(rel_path: str, exclude_dirs: tuple[str, ...]) -> str | None:
    excluded = {d.lower() for d in exclude_dirs}
    for part in PurePosixPath(rel_path).parts[:-1]:
        if part.startswith("."):
            return "hidden folder"
        if part.lower() in excluded:
            return "excluded folder"
    return None


def collect_directory(
    root: Path, exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDED_DIRS
) -> tuple[list[SourceFile], list[dict[str, str]]]:
    """Read every Markdown file under root. Returns (files, skipped entries)."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise InvalidInput(f"{root} is not a folder")
    excluded = {d.lower() for d in exclude_dirs}
    files: list[SourceFile] = []
    skipped: list[dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root)
        kept = []
        for name in sorted(dirnames):
            rel = (rel_dir / name).as_posix()
            if name.startswith("."):
                continue
            if name.lower() in excluded:
                skipped.append({"path": rel + "/", "reason": "excluded folder"})
            elif (Path(dirpath) / name).is_symlink():
                skipped.append({"path": rel + "/", "reason": "symlinked folder"})
            else:
                kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            if not name.lower().endswith(MARKDOWN_SUFFIXES):
                continue
            path = Path(dirpath) / name
            rel = (rel_dir / name).as_posix()
            if path.is_symlink():
                skipped.append({"path": rel, "reason": "symlink"})
                continue
            if not path.resolve().is_relative_to(root):
                skipped.append({"path": rel, "reason": "resolves outside the import folder"})
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped.append({"path": rel, "reason": f"larger than {MAX_FILE_BYTES} bytes"})
                continue
            text, _warnings = _decode(path.read_bytes())
            files.append(SourceFile(rel, text))
            if len(files) > MAX_FILES:
                raise InvalidInput(f"more than {MAX_FILES} Markdown files; import a smaller folder")
    return files, skipped


def validate_upload(path: str, content: str, exclude_dirs: tuple[str, ...]) -> tuple[str | None, str | None]:
    """(normalized path, None) for an acceptable uploaded file, else (None, reason)."""
    raw = path.replace("\\", "/").strip()
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return None, "path must be relative"
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None, "path must not contain '..'"
    norm = "/".join(parts)
    if not norm.lower().endswith(MARKDOWN_SUFFIXES):
        return None, "not a Markdown file"
    reason = _excluded(norm, exclude_dirs)
    if reason:
        return None, reason
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        return None, f"larger than {MAX_FILE_BYTES} bytes"
    return norm, None


# ---------------------------------------------------------------- parsing

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.M)


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "…"
    if isinstance(value, dict):
        return {str(k)[:100]: _json_safe(v, depth + 1) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in list(value)[:100]]
    if isinstance(value, str):
        return sanitize_text(value)[:5000]
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)[:500]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str, list[str]]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text, []
    body = text[match.end():]
    raw = match.group(1)
    if len(raw) > 64_000:
        return {}, body, ["frontmatter larger than 64 KB; ignored"]
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}, body, ["frontmatter is not valid YAML; ignored"]
    if data is None:
        return {}, body, []
    if not isinstance(data, dict):
        return {}, body, ["frontmatter is not a key/value map; ignored"]
    return _json_safe(data), body, []


def _folder_slug(rel_path: str) -> str | None:
    parts = PurePosixPath(rel_path).parts
    if len(parts) < 2:
        return None
    slug = re.sub(r"^\d+[\s._-]*", "", parts[0]).strip().lower()
    return re.sub(r"\s+", "-", slug) or None


def route_for(frontmatter: dict[str, Any], rel_path: str) -> tuple[str, str]:
    fm_type = str(frontmatter.get("type") or "").strip().lower()
    if fm_type in TYPE_ROUTES:
        return TYPE_ROUTES[fm_type]
    folder = _folder_slug(rel_path)
    if folder in FOLDER_ROUTES:
        return FOLDER_ROUTES[folder]
    return ("inbox", "note")


def _valid_date(text: str) -> str | None:
    text = text.strip()
    if not DATE_RE.match(text):
        return None
    try:
        dt.date.fromisoformat(text[:10])
    except ValueError:
        return None
    return text[:10] if len(text) == 10 else text


def occurred_at_for(frontmatter: dict[str, Any], rel_path: str) -> str | None:
    for key in ("date", "occurred_at", "due", "created"):
        value = frontmatter.get(key)
        if isinstance(value, str) and (date := _valid_date(value)):
            return date
    name = PurePosixPath(rel_path).stem
    if m := re.search(r"(\d{4}-\d{2}-\d{2})", name):
        return _valid_date(m.group(1))
    if m := re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", name):
        return _valid_date(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return None


def title_for(frontmatter: dict[str, Any], body: str, rel_path: str) -> str:
    title = frontmatter.get("title")
    if not isinstance(title, str) or not title.strip():
        m = _H1_RE.search(body)
        title = m.group(1) if m else PurePosixPath(rel_path).stem
    title = WIKILINK_RE.sub(lambda m: m.group(3) or m.group(1), sanitize_text(title)).strip()
    return title[:300] or PurePosixPath(rel_path).stem


def wikilink_targets(body: str) -> list[str]:
    seen: list[str] = []
    for m in WIKILINK_RE.finditer(body):
        target = (m.group(1) or "").strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def _fm_tags(frontmatter: dict[str, Any]) -> list[str]:
    tags = frontmatter.get("tags")
    if isinstance(tags, str):
        return re.split(r"[,\s]+", tags)
    if isinstance(tags, list):
        return [str(t) for t in tags]
    return []


# ---------------------------------------------------------------- import


def _strip_suffix(path: str) -> str:
    lower = path.lower()
    for suffix in MARKDOWN_SUFFIXES:
        if lower.endswith(suffix):
            return path[: -len(suffix)]
    return path


def _mark_synced(conn: sqlite3.Connection, record_id: str, unresolved: list[str] | None = None) -> None:
    """Remember the record's updated_at right after an import write; a later difference means
    the user edited it inside CORTEX."""
    conn.execute(
        "UPDATE records SET metadata = json_set(metadata, '$.import.synced_at', updated_at) WHERE id = ?",
        (record_id,),
    )
    if unresolved is not None:
        conn.execute(
            "UPDATE records SET metadata = json_set(metadata, '$.import.unresolved_links', json(?)) WHERE id = ?",
            (json.dumps(unresolved), record_id),
        )


def import_markdown(
    conn: sqlite3.Connection,
    files: list[SourceFile],
    options: ImportOptions,
    skipped: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if options.route not in ROUTES:
        raise InvalidInput(f"route must be one of {', '.join(ROUTES)}")
    if len(files) > MAX_FILES:
        raise InvalidInput(f"more than {MAX_FILES} files")
    label = re.sub(r"[^\w .\-]", "", options.source_label).strip()[:80] or "import"
    if options.project_id:
        project = conn.execute(
            "SELECT 1 FROM records WHERE id = ? AND kind = 'project' AND deleted_at IS NULL", (options.project_id,)
        ).fetchone()
        if project is None:
            raise InvalidInput("project_id must reference an existing project record")

    batch_id = new_id()
    now = utcnow()
    counts: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    touched: dict[str, tuple[str, list[str]]] = {}  # record id -> (path, link targets)
    link_stats = {"resolved": 0, "unresolved": 0}
    report: dict[str, Any] = {
        "batch_id": None if options.dry_run else batch_id,
        "dry_run": options.dry_run,
        "route": options.route,
        "source": label,
        "files": entries,
        "skipped": list(skipped or []),
        "counts": counts,
        "links": link_stats,
        "conflicts": None,
    }
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO import_batches (id, source, route, created_at) VALUES (?, ?, ?, ?)",
                (batch_id, options.root or label, options.route, now),
            )
            seen_paths: set[str] = set()
            for f in files:
                if f.path in seen_paths:
                    counts["skipped"] += 1
                    report["skipped"].append({"path": f.path, "reason": "duplicate path in this import"})
                    continue
                seen_paths.add(f.path)
                entry = _import_one(conn, f, options, label, batch_id, now)
                counts[entry["action"]] += 1
                entries.append(entry)
                if entry["action"] in ("created", "updated"):
                    touched[entry["record_id"]] = (f.path, entry.pop("_links"))
                else:
                    entry.pop("_links", None)

            # Resolve wikilinks against every record imported from this source.
            by_path: dict[str, str] = {}
            by_stem: dict[str, set[str]] = defaultdict(set)
            for row in conn.execute(
                "SELECT id, source_ref FROM records WHERE deleted_at IS NULL AND source_ref LIKE ? ESCAPE '\\'",
                (label.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%",),
            ):
                key = _strip_suffix(row["source_ref"][len(label) + 1:]).lower()
                by_path[key] = row["id"]
                by_stem[PurePosixPath(key).name].add(row["id"])
            for record_id, (_path, targets) in touched.items():
                conn.execute(
                    "DELETE FROM relationships WHERE source_id = ? AND origin = 'import' AND rel_type = 'references'",
                    (record_id,),
                )
                unresolved = []
                for target in targets:
                    key = _strip_suffix(target.replace("\\", "/").lstrip("./")).lower()
                    target_id = by_path.get(key)
                    if target_id is None and len(by_stem.get(PurePosixPath(key).name, ())) == 1:
                        target_id = next(iter(by_stem[PurePosixPath(key).name]))
                    if target_id is None:
                        unresolved.append(target)
                        continue
                    link_stats["resolved"] += 1
                    if target_id != record_id:
                        create_relationship(conn, record_id, target_id, "references", origin="import", log=False)
                link_stats["unresolved"] += len(unresolved)
                _mark_synced(conn, record_id, unresolved)

            summary = ", ".join(f"{n} {action}" for action, n in sorted(counts.items())) or "nothing"
            conn.execute(
                "UPDATE import_batches SET stats = ? WHERE id = ?",
                (json.dumps({"counts": counts, "links": link_stats}), batch_id),
            )
            log_activity(conn, "imported", None, label, f"{summary} ({options.route})")
            if options.dry_run:
                raise _DryRun
    except _DryRun:
        pass
    report["counts"] = dict(counts)
    if not options.dry_run and touched:
        found = conflicts.detect(conn, record_ids=list(touched))
        report["conflicts"] = {"new": len(found["new_conflicts"])}
    return report


def _import_one(
    conn: sqlite3.Connection, f: SourceFile, options: ImportOptions, label: str, batch_id: str, now: str
) -> dict[str, Any]:
    content = sanitize_text(f.content)
    frontmatter, body, warnings = parse_frontmatter(content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    source_ref = f"{label}/{f.path}"
    suggested_category, kind = route_for(frontmatter, f.path)
    category = suggested_category if options.route == "by_type" else "inbox"
    title = title_for(frontmatter, body, f.path)
    links = wikilink_targets(body)
    tags = normalize_tags(["imported", *(t for t in [_folder_slug(f.path)] if t), *_fm_tags(frontmatter), *options.extra_tags])
    metadata: dict[str, Any] = {
        "frontmatter": frontmatter,
        "import": {"source": label, "path": f.path, "sha256": digest, "imported_at": now, "batch_id": batch_id},
    }
    if options.root:
        metadata["import"]["root"] = options.root
    if category == "inbox":
        metadata["suggested_category"] = suggested_category
    status = frontmatter.get("status")
    fields: dict[str, Any] = {
        "title": title,
        "content": body,
        "category": category,
        "kind": kind,
        "status": str(status) if status not in (None, "") else None,
        "source_ref": source_ref,
        "occurred_at": occurred_at_for(frontmatter, f.path),
        "tags": tags,
        "metadata": metadata,
    }
    if options.project_id and kind != "project":
        fields["project_id"] = options.project_id
    entry = {"path": f.path, "title": title, "category": category, "kind": kind, "warnings": warnings, "_links": links}

    existing = find_by_source_ref(conn, source_ref)
    if existing is None:
        entry["record_id"] = create_record(conn, fields, import_batch_id=batch_id, source_hash=digest, log=False)
        entry["action"] = "created"
        return entry

    entry["record_id"] = existing["id"]
    entry["category"] = existing["category"]
    if existing["deleted_at"] is not None:
        entry["action"] = "skipped-trashed"
        return entry
    if existing["source_hash"] == digest:
        entry["action"] = "unchanged"
        return entry
    old_meta = json.loads(existing["metadata"])
    if existing["updated_at"] != old_meta.get("import", {}).get("synced_at") and not options.update_edited:
        entry["action"] = "skipped-local-edits"
        return entry
    # Keep the user's own triage (category/kind/project) and metadata keys; refresh the rest.
    merged_meta = {**old_meta, "frontmatter": frontmatter, "import": {**old_meta.get("import", {}), **metadata["import"]}}
    patch = {
        "title": title,
        "content": body,
        "status": fields["status"],
        "occurred_at": fields["occurred_at"],
        "metadata": merged_meta,
        "tags": normalize_tags([*record_tags(conn, existing["id"]), *tags]),
    }
    update_record(conn, existing["id"], patch, reason=f"re-import {batch_id}", source_hash=digest, log=False)
    entry["action"] = "updated"
    return entry


def ensure_project(conn: sqlite3.Connection, name: str) -> str:
    """Id of the active project with this title, creating it if needed."""
    row = conn.execute(
        "SELECT id FROM records WHERE kind = 'project' AND deleted_at IS NULL AND lower(title) = lower(?)", (name,)
    ).fetchone()
    if row:
        return row["id"]
    return create_record(conn, {"title": name, "category": "projects", "kind": "project", "status": "active"})
