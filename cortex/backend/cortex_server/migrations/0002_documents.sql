-- PDF documents. A PDF is represented by an ordinary record (title, tags, links, trash, history);
-- this table adds what was read from the file, and document_pages holds the text of each page
-- so passages can be searched. The PDF itself is never copied or modified: `path` points at the
-- file that was imported. Page text is derived data and is replaced whenever the file changes.

CREATE TABLE documents (
    record_id    TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    bytes        INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    page_count   INTEGER NOT NULL DEFAULT 0,
    text_pages   INTEGER NOT NULL DEFAULT 0,
    failed_pages INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL CHECK (status IN ('ok', 'partial', 'no_text', 'failed')),
    error        TEXT,
    pdf_meta     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(pdf_meta)),
    extractor    TEXT NOT NULL,
    extracted_at TEXT NOT NULL
);

-- Only pages that yielded text are stored; `page` is 1-based.
CREATE TABLE document_pages (
    id        INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL REFERENCES documents(record_id) ON DELETE CASCADE,
    page      INTEGER NOT NULL,
    text      TEXT NOT NULL,
    UNIQUE (record_id, page)
);

-- External-content index over document_pages, kept in step by triggers (they also fire for
-- rows removed by the ON DELETE CASCADE above, e.g. when a record is purged).
CREATE VIRTUAL TABLE document_pages_fts USING fts5(
    text,
    content = 'document_pages',
    content_rowid = 'id',
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER document_pages_ai AFTER INSERT ON document_pages BEGIN
    INSERT INTO document_pages_fts (rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER document_pages_ad AFTER DELETE ON document_pages BEGIN
    INSERT INTO document_pages_fts (document_pages_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER document_pages_au AFTER UPDATE ON document_pages BEGIN
    INSERT INTO document_pages_fts (document_pages_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO document_pages_fts (rowid, text) VALUES (new.id, new.text);
END;
