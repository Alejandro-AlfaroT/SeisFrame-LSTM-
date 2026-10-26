# CORTEX

A local knowledge base for the SeisFrame project: notes, facts, decisions, milestones and work
sessions, stored in one SQLite file on this machine and shown as a brain-shaped particle cloud
you can navigate. Nothing is sent to external services.

The brain is a navigation metaphor. Each memory category is drawn in one region:

| Region (visual) | Category | Holds |
|---|---|---|
| Hippocampus | Facts | Stable knowledge with source, confidence and verification state |
| Occipital | Sessions | Dated work sessions, daily notes, reviews |
| Temporal | Inbox | New captures and notes waiting for triage |
| Frontal | Conflicts | Candidate contradictions between records, for review |
| Prefrontal | Projects | Projects, decisions, milestones, plans, experiments, tasks |

These mappings organize the interface. They are not claims about what those brain areas do.

## Running it

### Setting up a Windows machine

From the repository root, in the PyCharm terminal with the OpPy environment (any Python 3.11+):

```
python cortex/setup_cortex.py
```

It creates `backend/.venv`, installs and builds the interface (offering to install Node.js
with winget if it is missing), imports the SeisFrame vault from OneDrive (notes, and the text of
its PDFs), puts a CORTEX shortcut on the desktop and in the Start menu, and starts CORTEX. Running
it again is safe: it refreshes the build and re-imports only notes and PDFs that changed. The
first import reads all the PDFs, which takes a minute or two. Options: `--vault "<folder>"` if the vault is not in
OneDrive, `--skip-import`, `--no-shortcut`, `--no-start`, `--yes`.

### Day to day

Double-click CORTEX on the desktop (or search for it in Start). `Start CORTEX.cmd` starts the
server in a minimized "CORTEX server" window unless one is already running, waits for it, and opens
http://127.0.0.1:8420 in the default browser. Close the "CORTEX server" window to stop it. By hand,
from `cortex/`:

```
backend/.venv/Scripts/python.exe backend/run.py serve
```

For UI development, run the API as above and `npm run dev` in `frontend/` instead, then open
http://127.0.0.1:5173 (Vite proxies `/api` to port 8420). `.claude/launch.json` in the repository has
`cortex-api` and `cortex-ui` entries for the same two servers.

The database lives at `cortex/data/cortex.db` (ignored by git, as are its backups). Use `--db <file>`
or `CORTEX_DB` for another one.

## Importing the SeisFrame vault

```
backend/.venv/Scripts/python.exe backend/run.py import-md "C:/Users/andro/OneDrive/SeisFrame-Obsidian-Starter" --route by_type --project "SeisFrame V6" --dry-run
```

Drop `--dry-run` to write. `setup_cortex.py` runs this command (without `--dry-run`) on the vault
it finds in OneDrive; `Templates/` is skipped. The UI's Import panel does the same for picked files
or a picked folder, always previewing first. Each machine keeps its own database, so import on each.

- `--route inbox` (default) puts every note in the Inbox with a suggested category, for triage.
  `--route by_type` files notes directly using their frontmatter `type` (decision → Projects,
  checkpoint → milestone, daily / research-session → Sessions, literature / concept → Facts…),
  falling back to the folder name (`06 Decisions`, `07 Daily`, …).
- Titles come from the first `#` heading, dates from `date` / `due` / `created` or the file name,
  status and tags from frontmatter. `[[wikilinks]]` become `references` relationships.
- Re-importing is safe: unchanged files are skipped; files changed on disk update their record
  (the old version stays in its history); records you edited in CORTEX since the import, or put
  in the trash, are left alone unless you pass `--update-edited`.
- Nothing imported is marked verified.
- **PDFs** in the folder (the papers in `08 Assets/Literature/`) each become a record (kind `pdf`,
  filed under Facts with `by_type`), and the text of every page is stored so it can be searched.
  A note's `[[…/paper.pdf]]` link becomes a `references` relationship to the PDF's record. The PDF
  itself stays where it is and is never changed. Only new or changed PDFs are read again
  (same size and date, or same SHA-256 → skipped). `--no-pdfs` leaves PDFs out.
- What PDF reading cannot do: scanned PDFs have no text layer and CORTEX does not run OCR, so
  their contents are not searchable (in the vault, ASCE 7-22 is such a scan; ACI 318-25 is
  readable). Some older journal PDFs use font ligatures that come out garbled, e.g. "sti/VTness"
  for "stiffness" in Ibarra 2005, so search misses those words there. Pages that fail to parse
  (one figure page of Meng 2026) are skipped and counted. The import summary lists every PDF that
  was only partly readable.

## What it does

- **Records** have an id, title, Markdown content, category, kind, status, timestamps
  (created, updated, and the date the record is about), tags, project, source reference,
  confidence, verification, an optional subject / predicate / value triple, metadata, and typed
  relationships (`relates_to`, `references`, `supports`, `contradicts`, `depends_on`, `part_of`,
  `supersedes`, `derived_from`, `decided_in`, `follows`).
- **Capture** forms for notes, facts, decisions (with rationale and alternatives), session
  summaries (with activities and related documents), projects (with objectives) and milestones.
  Any capture can link existing records.
- **Search**: SQLite FTS5 with bm25 ranking and highlighted snippets; bare words are prefix
  matches, `"quotes"` are phrases, `-word` excludes. Filters for category, project, tags, date
  range, verified-only and demo data. Matching records light up in the brain.
- **Search inside PDFs**: the same query also runs over the pages of imported PDFs (Search →
  "In PDFs"), listing each matching paper with its best three passages and page numbers. A
  passage opens the PDF's record at that page, showing the extracted text with the words marked,
  "find in this PDF" with previous / next matching page, and a link that opens the original PDF
  at that page in the browser's own PDF viewer.
- **Similar records** use TF-IDF cosine similarity over words (for a PDF, its title, record and
  first two pages). This is lexical, not a neural-embedding semantic search; none is
  implemented, and nothing calls an external model.
- **Knowledge graph**: relationships show as arcs inside the brain and as a force-directed graph
  in the Graph panel, where a link's type can be changed or the link removed. The record panel
  edits relationships too.
- **Conflict detection** proposes candidates; it never edits a record. It flags facts with the
  same subject and predicate but different values, and near-identical statements in different
  records that differ in a number, a negation, or an opposing term (increase / decrease,
  retains / removes, …). To avoid false alarms, it does not compare statements that name
  different sources or things (dates, IDs, link targets, URLs, file paths, inline code, citation
  years like "(2024)"), numbers in records dated on different days, or lines that recur in four or
  more records. On the SeisFrame vault (176 notes as of 2026-10-23) it finds no candidates. In
  review you prefer one side, keep both, or dismiss; preferring a side adds a
  `supersedes` link and a marker on the other record, and reopening undoes both.
- **Dashboard**: total records, verified facts, active projects, sessions, open conflicts, inbox
  size, and recent activity.
- **Views**: free 3D orbit plus axial, sagittal and coronal camera presets, and three flat
  projections of the same particles as thumbnails.

## Keeping data safe

- Every edit stores the previous state in `record_versions`; any version can be restored.
- Saving checks the record's `updated_at` against what the editor loaded and refuses to
  overwrite a newer change (HTTP 409).
- Delete moves a record to the trash. Permanent deletion is a separate step in Data → Trash and
  needs typed confirmation.
- Schema changes go through numbered migrations (`backend/cortex_server/migrations/`), each in a
  transaction; a database with data is backed up before any migration runs.
- Backups (`backend/run.py backup` or Data → Back up now) use SQLite's online backup API into
  `data/backups/`. Restore with the server stopped:
  `backend/run.py restore <backup file> --yes` (the current database is backed up first).
- Exports: everything as JSON (including trash and version history), or active records as a zip
  of Markdown files with YAML frontmatter. The JSON export lists the PDFs but leaves out their
  extracted page text (re-import the PDFs to re-create it); database backups include it.

## Security

- The server binds to 127.0.0.1 only and refuses other hosts (`TrustedHostMiddleware`).
- Every write needs an `X-Cortex-Client: 1` header, which other web pages cannot send without a
  CORS preflight, so a page open in your browser cannot change the database.
- Imported files are untrusted: only `.md` files up to 2 MB; hidden folders, symlinks and paths
  leaving the chosen folder are skipped; YAML is parsed with `safe_load` and size-limited.
  Markdown renders without raw HTML, `javascript:` and other non-web links are dropped, and
  remote images are never loaded. The built UI ships a Content-Security-Policy that limits it to
  its own origin.
- PDFs are untrusted too. Each is parsed by pypdf in a separate Python process
  (`pdf_worker.py`, started in isolated mode) that is killed after 300 s (`CORTEX_PDF_TIMEOUT`),
  so a malformed file cannot hang or crash CORTEX; PDFs over 512 MB are skipped. Only text and the
  title / author fields are kept; nothing embedded in a PDF runs. PDFs that need a password are
  not opened. The API serves a PDF file only if an import registered it, from where it was
  imported.

## Code layout

| Path | Contents |
|---|---|
| `backend/cortex_server/db.py` | Connections, transactions, migrations, backups |
| `backend/cortex_server/repository.py` | Records, tags, versions, trash, relationships, stats |
| `backend/cortex_server/search.py` | FTS5 query building and filters; TF-IDF similarity |
| `backend/cortex_server/conflicts.py` | Candidate detection, review, resolution |
| `backend/cortex_server/importer.py` | Markdown / Obsidian import, including a folder's PDFs |
| `backend/cortex_server/documents.py`, `pdf_worker.py` | PDF text extraction (child process), page storage and lookup |
| `backend/cortex_server/brain.py` | Region clusters and graph data for the UI |
| `backend/cortex_server/backup.py` | Backups, restore, JSON and Markdown export |
| `backend/cortex_server/demo.py` | Fictional demonstration records (load / remove from Data) |
| `backend/cortex_server/app.py`, `cli.py` | HTTP API and command line |
| `frontend/src/brain/` | Point-cloud generation (signed-distance brain model), 3D view, slice thumbnails |
| `frontend/src/components/` | Panels: dashboard, regions, record, search, capture, import, conflicts, graph, data |

## Tests

```
cd backend && .venv/Scripts/python.exe -m pytest
cd frontend && npm test
```

The backend suite (pytest) covers migrations and rollback, CRUD and validation, version history,
stale-write refusal, trash and purge, full-text search including hostile FTS syntax, every
conflict rule and false-positive guard, resolution and reopening, import routing / links /
re-import / untrusted input, PDF extraction (scanned, broken, password-protected and
owner-locked files, the time limit), passage search, re-reading only changed PDFs, purging a PDF's
text, the HTTP API and its write guard, backups, exports, restore, and data surviving both an app
restart and a killed-and-restarted server process. The tests build their PDFs byte by byte. The
frontend suite (Vitest) covers the brain model (determinism, regions, fissure, placement), safe
Markdown rendering, the API client, capture payloads, the conflict-review panel, PDF passages in
search and the PDF page reader.

## Demonstration data

Data → Load demo data adds eleven fictional records about a museum exhibit, unrelated to
SeisFrame, flagged `DEMO`, with two deliberate contradictions. Data → Remove demo data deletes
only those records. The SeisFrame database was created without them.
