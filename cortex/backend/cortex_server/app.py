"""HTTP API (FastAPI). Binds to localhost; see cli.py `serve`."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, backup, brain, conflicts, demo, importer, repository, search
from .config import Settings
from .db import connect, migrate, schema_version, transaction, utcnow
from .repository import InvalidInput, NotFound, StaleWrite

MAX_BODY_BYTES = 64 * 1024 * 1024
CLIENT_HEADER = "x-cortex-client"
DEV_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]
Category = Literal["facts", "sessions", "inbox", "projects"]


class LinkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str
    rel_type: str = "relates_to"
    direction: Literal["outgoing", "incoming"] = "outgoing"
    note: str | None = None


class RecordCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(default="", max_length=2_000_000)
    category: Category
    kind: str = "note"
    status: str | None = None
    project_id: str | None = None
    source_ref: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    verified: bool = False
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None
    occurred_at: str | None = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    links: list[LinkSpec] = []


class RecordPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=500)
    content: str | None = Field(default=None, max_length=2_000_000)
    category: Category | None = None
    kind: str | None = None
    status: str | None = None
    project_id: str | None = None
    source_ref: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    verified: bool | None = None
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None
    occurred_at: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    expected_updated_at: str | None = None


class RelationshipIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    target_id: str
    rel_type: str
    note: str | None = None


class RelationshipPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rel_type: str | None = None
    note: str | None = None


class RevertIn(BaseModel):
    version_id: int


class ConflictFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_a_id: str
    record_b_id: str
    excerpt_a: str = ""
    excerpt_b: str = ""
    explanation: str = ""


class ConflictResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution: Literal["prefer_a", "prefer_b", "keep_both", "not_a_conflict"]
    note: str | None = Field(default=None, max_length=5000)


class UploadFile(BaseModel):
    path: str = Field(max_length=1000)
    content: str


class MarkdownImport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    files: list[UploadFile] = Field(max_length=importer.MAX_FILES)
    route: Literal["inbox", "by_type"] = "inbox"
    project_id: str | None = None
    source_label: str = Field(default="upload", max_length=80)
    dry_run: bool = True
    include_templates: bool = False
    update_edited: bool = False


class BackupIn(BaseModel):
    label: str = Field(default="manual", max_length=40)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    migrate(settings.db_path, settings.backup_dir)

    app = FastAPI(title="CORTEX", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-Cortex-Client"],
    )

    @app.middleware("http")
    async def guard_writes(request: Request, call_next):  # type: ignore[no-untyped-def]
        # A custom header cannot be sent cross-site without a CORS preflight, so other web
        # pages open in the browser cannot trigger writes against this local server.
        if request.url.path.startswith("/api/") and request.method in ("POST", "PATCH", "PUT", "DELETE"):
            if request.headers.get(CLIENT_HEADER) != "1":
                return JSONResponse({"detail": "missing X-Cortex-Client header"}, status_code=403)
            length = request.headers.get("content-length")
            if length and int(length) > MAX_BODY_BYTES:
                return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(NotFound)
    async def not_found(_request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(InvalidInput)
    async def invalid(_request: Request, exc: InvalidInput) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(StaleWrite)
    async def stale(_request: Request, exc: StaleWrite) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def db() -> Iterator[sqlite3.Connection]:
        conn = connect(settings.db_path)
        try:
            yield conn
        finally:
            conn.close()

    Conn = Depends(db)

    # ------------------------------------------------------------ overview

    @app.get("/api/health")
    def health(conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "schema_version": schema_version(conn),
            "database": str(settings.db_path),
        }

    @app.get("/api/stats")
    def stats(conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        backups = backup.list_backups(settings.backup_dir)
        return {**repository.stats(conn), "last_backup": backups[0]["name"] if backups else None}

    @app.get("/api/activity")
    def activity(limit: int = Query(30, ge=1, le=200), conn: sqlite3.Connection = Conn) -> list[dict[str, Any]]:
        return repository.recent_activity(conn, limit)

    @app.get("/api/tags")
    def tags(conn: sqlite3.Connection = Conn) -> list[dict[str, Any]]:
        return repository.list_tags(conn)

    @app.get("/api/projects")
    def projects(conn: sqlite3.Connection = Conn) -> list[dict[str, Any]]:
        return repository.list_projects(conn)

    @app.get("/api/relationship-types")
    def relationship_types() -> list[str]:
        return list(repository.REL_TYPES)

    @app.get("/api/brain")
    def brain_map(include_demo: bool = True, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        return brain.brain_map(conn, include_demo)

    @app.get("/api/graph")
    def graph(
        record_id: str | None = None,
        depth: int = Query(2, ge=0, le=4),
        limit: int = Query(150, ge=1, le=1000),
        conn: sqlite3.Connection = Conn,
    ) -> dict[str, Any]:
        return brain.graph(conn, record_id, depth, limit)

    # ------------------------------------------------------------ search

    @app.get("/api/search")
    def run_search(
        q: str = "",
        mode: Literal["fulltext", "similar"] = "fulltext",
        category: list[str] = Query(default=[]),
        kind: list[str] = Query(default=[]),
        tag: list[str] = Query(default=[]),
        project_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        verified: bool | None = None,
        include_demo: bool = True,
        status: str | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        conn: sqlite3.Connection = Conn,
    ) -> dict[str, Any]:
        filters = search.Filters(
            categories=category, kinds=kind, project_id=project_id, tags=tag, date_from=date_from,
            date_to=date_to, verified=verified, include_demo=include_demo, status=status,
        )
        if mode == "similar":
            return search.similar(conn, text=q, filters=filters, limit=limit)
        return search.search(conn, q, filters, limit, offset)

    @app.get("/api/resolve")
    def resolve(ref: str, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        found = repository.resolve_reference(conn, ref)
        if found is None:
            raise HTTPException(404, f"no record matches '{ref}'")
        return found

    # ------------------------------------------------------------ records

    @app.post("/api/records", status_code=201)
    def create_record(body: RecordCreate, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        data = body.model_dump(exclude={"links"})
        with transaction(conn):
            record_id = repository.create_record(conn, data)
            for link in body.links:
                source, target = (record_id, link.target_id) if link.direction == "outgoing" else (link.target_id, record_id)
                repository.create_relationship(conn, source, target, link.rel_type, link.note)
        if body.category == "facts" and body.subject and body.predicate:
            conflicts.detect(conn, record_ids=[record_id])
        return repository.get_record(conn, record_id)

    @app.get("/api/records/{record_id}")
    def get_record(record_id: str, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        return repository.get_record(conn, record_id)

    @app.patch("/api/records/{record_id}")
    def update_record(record_id: str, body: RecordPatch, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        patch = body.model_dump(exclude_unset=True)
        expected = patch.pop("expected_updated_at", None)
        for key in ("title", "content", "category", "kind", "verified", "tags", "metadata"):
            if key in patch and patch[key] is None:
                del patch[key]  # not nullable; an explicit null means "leave unchanged"
        repository.update_record(conn, record_id, patch, expected_updated_at=expected)
        return repository.get_record(conn, record_id)

    @app.delete("/api/records/{record_id}")
    def trash_record(record_id: str, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        repository.trash_record(conn, record_id)
        return repository.get_record(conn, record_id)

    @app.post("/api/records/{record_id}/restore")
    def restore_record(record_id: str, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        repository.restore_record(conn, record_id)
        return repository.get_record(conn, record_id)

    @app.delete("/api/records/{record_id}/purge")
    def purge_record(record_id: str, confirm: str = "", conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        if confirm != record_id:
            raise InvalidInput("permanent deletion needs ?confirm=<record id>")
        repository.purge_record(conn, record_id)
        return {"purged": record_id}

    @app.get("/api/records/{record_id}/versions")
    def versions(record_id: str, conn: sqlite3.Connection = Conn) -> list[dict[str, Any]]:
        return repository.record_versions(conn, record_id)

    @app.post("/api/records/{record_id}/revert")
    def revert(record_id: str, body: RevertIn, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        repository.revert_record(conn, record_id, body.version_id)
        return repository.get_record(conn, record_id)

    @app.get("/api/records/{record_id}/similar")
    def similar(record_id: str, limit: int = Query(8, ge=1, le=50), conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        repository.get_record(conn, record_id)
        return search.similar(conn, record_id=record_id, limit=limit)

    @app.get("/api/trash")
    def trash(conn: sqlite3.Connection = Conn) -> list[dict[str, Any]]:
        return repository.list_trash(conn)

    # ------------------------------------------------------------ relationships

    @app.post("/api/relationships", status_code=201)
    def create_relationship(body: RelationshipIn, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        rel_id = repository.create_relationship(conn, body.source_id, body.target_id, body.rel_type, body.note)
        return {"id": rel_id}

    @app.patch("/api/relationships/{rel_id}")
    def update_relationship(rel_id: str, body: RelationshipPatch, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        repository.update_relationship(conn, rel_id, body.rel_type, body.note)
        return {"id": rel_id}

    @app.delete("/api/relationships/{rel_id}")
    def delete_relationship(rel_id: str, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        repository.delete_relationship(conn, rel_id)
        return {"deleted": rel_id}

    # ------------------------------------------------------------ conflicts

    @app.get("/api/conflicts")
    def list_conflicts(
        status: Literal["open", "resolved", "dismissed"] | None = None,
        include_demo: bool = True,
        conn: sqlite3.Connection = Conn,
    ) -> list[dict[str, Any]]:
        return conflicts.list_conflicts(conn, status, include_demo)

    @app.get("/api/conflicts/{conflict_id}")
    def get_conflict(conflict_id: str, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        return conflicts.get_conflict(conn, conflict_id)

    @app.post("/api/conflicts/scan")
    def scan_conflicts(conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        return conflicts.detect(conn)

    @app.post("/api/conflicts", status_code=201)
    def flag_conflict(body: ConflictFlag, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        conflict_id = conflicts.flag_conflict(
            conn, body.record_a_id, body.record_b_id, body.excerpt_a, body.excerpt_b, body.explanation
        )
        return conflicts.get_conflict(conn, conflict_id)

    @app.post("/api/conflicts/{conflict_id}/resolve")
    def resolve_conflict(conflict_id: str, body: ConflictResolution, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        conflicts.resolve_conflict(conn, conflict_id, body.resolution, body.note)
        return conflicts.get_conflict(conn, conflict_id)

    @app.post("/api/conflicts/{conflict_id}/reopen")
    def reopen_conflict(conflict_id: str, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        conflicts.reopen_conflict(conn, conflict_id)
        return conflicts.get_conflict(conn, conflict_id)

    # ------------------------------------------------------------ import / export / backup

    @app.post("/api/import/markdown")
    def import_markdown(body: MarkdownImport, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        exclude = () if body.include_templates else importer.DEFAULT_EXCLUDED_DIRS
        files, skipped = [], []
        for upload in body.files:
            path, reason = importer.validate_upload(upload.path, upload.content, exclude)
            if path is None:
                skipped.append({"path": upload.path, "reason": reason or "rejected"})
            else:
                files.append(importer.SourceFile(path, upload.content))
        options = importer.ImportOptions(
            route=body.route, project_id=body.project_id, source_label=body.source_label,
            dry_run=body.dry_run, update_edited=body.update_edited, exclude_dirs=exclude,
        )
        return importer.import_markdown(conn, files, options, skipped)

    @app.get("/api/imports")
    def imports(conn: sqlite3.Connection = Conn) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute("SELECT * FROM import_batches ORDER BY created_at DESC LIMIT 100")]

    @app.post("/api/backup", status_code=201)
    def make_backup(body: BackupIn | None = None, conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        result = backup.create_backup(conn, settings.backup_dir, (body or BackupIn()).label)
        repository.log_activity(conn, "backup", None, result["name"], None)
        return result

    @app.get("/api/backups")
    def backups() -> list[dict[str, Any]]:
        return backup.list_backups(settings.backup_dir)

    @app.get("/api/export/json")
    def export_json(conn: sqlite3.Connection = Conn) -> JSONResponse:
        stamp = utcnow()[:19].replace(":", "")
        return JSONResponse(
            backup.export_json(conn),
            headers={"Content-Disposition": f'attachment; filename="cortex-export-{stamp}.json"'},
        )

    @app.get("/api/export/markdown")
    def export_markdown(conn: sqlite3.Connection = Conn) -> Response:
        stamp = utcnow()[:19].replace(":", "")
        return Response(
            backup.export_markdown_zip(conn),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="cortex-markdown-{stamp}.zip"'},
        )

    # ------------------------------------------------------------ demo data

    @app.post("/api/demo")
    def load_demo(conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        return demo.load_demo(conn)

    @app.delete("/api/demo")
    def purge_demo(conn: sqlite3.Connection = Conn) -> dict[str, Any]:
        return demo.purge_demo(conn)

    # ------------------------------------------------------------ built frontend

    dist: Path = settings.frontend_dist
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
