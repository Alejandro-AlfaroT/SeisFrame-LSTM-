-- CORTEX initial schema.
-- Records are never hard-deleted by normal operations: deletion moves a record to the trash
-- (deleted_at), every edit snapshots the previous state into record_versions, and conflict
-- resolution only adds relationships and metadata.

CREATE TABLE import_batches (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    route       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    stats       TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(stats))
);

CREATE TABLE records (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL CHECK (length(trim(title)) > 0),
    content         TEXT NOT NULL DEFAULT '',
    category        TEXT NOT NULL CHECK (category IN ('facts', 'sessions', 'inbox', 'projects')),
    kind            TEXT NOT NULL DEFAULT 'note',
    status          TEXT,
    project_id      TEXT REFERENCES records(id) ON DELETE SET NULL,
    source_ref      TEXT,
    source_hash     TEXT,
    confidence      REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    verified        INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    verified_at     TEXT,
    subject         TEXT,
    predicate       TEXT,
    value           TEXT,
    occurred_at     TEXT,
    metadata        TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata)),
    is_demo         INTEGER NOT NULL DEFAULT 0 CHECK (is_demo IN (0, 1)),
    import_batch_id TEXT REFERENCES import_batches(id) ON DELETE SET NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT
);

CREATE INDEX idx_records_category   ON records(category) WHERE deleted_at IS NULL;
CREATE INDEX idx_records_kind       ON records(kind);
CREATE INDEX idx_records_project    ON records(project_id);
CREATE INDEX idx_records_occurred   ON records(occurred_at);
CREATE INDEX idx_records_updated    ON records(updated_at);
CREATE INDEX idx_records_source_ref ON records(source_ref);
CREATE INDEX idx_records_subject    ON records(lower(subject), lower(predicate)) WHERE subject IS NOT NULL;

CREATE TABLE tags (
    record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    tag       TEXT NOT NULL,
    PRIMARY KEY (record_id, tag)
);
CREATE INDEX idx_tags_tag ON tags(tag);

CREATE TABLE relationships (
    id         TEXT PRIMARY KEY,
    source_id  TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    target_id  TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    rel_type   TEXT NOT NULL,
    note       TEXT,
    origin     TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (source_id <> target_id),
    UNIQUE (source_id, target_id, rel_type)
);
CREATE INDEX idx_rel_source ON relationships(source_id);
CREATE INDEX idx_rel_target ON relationships(target_id);

CREATE TABLE record_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id  TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    snapshot   TEXT NOT NULL CHECK (json_valid(snapshot)),
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_versions_record ON record_versions(record_id, id);

CREATE TABLE conflicts (
    id                         TEXT PRIMARY KEY,
    record_a_id                TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    record_b_id                TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    excerpt_a                  TEXT NOT NULL,
    excerpt_b                  TEXT NOT NULL,
    reason                     TEXT NOT NULL,
    explanation                TEXT NOT NULL,
    score                      REAL NOT NULL,
    fingerprint                TEXT NOT NULL UNIQUE,
    status                     TEXT NOT NULL DEFAULT 'open'
                               CHECK (status IN ('open', 'resolved', 'dismissed')),
    resolution                 TEXT CHECK (resolution IS NULL OR resolution IN
                               ('prefer_a', 'prefer_b', 'keep_both', 'not_a_conflict')),
    resolution_note            TEXT,
    resolution_relationship_id TEXT,
    detected_at                TEXT NOT NULL,
    resolved_at                TEXT,
    CHECK (record_a_id <> record_b_id)
);
CREATE INDEX idx_conflicts_status ON conflicts(status);

-- Activity survives purges on purpose, so it carries the title instead of a foreign key.
CREATE TABLE activity (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    action    TEXT NOT NULL,
    record_id TEXT,
    title     TEXT,
    detail    TEXT
);

-- Full-text index maintained by the application in the same transaction as each write.
CREATE VIRTUAL TABLE records_fts USING fts5(
    record_id UNINDEXED,
    title,
    content,
    tags,
    source_ref,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
