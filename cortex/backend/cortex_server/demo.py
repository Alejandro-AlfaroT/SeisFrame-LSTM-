"""Demonstration records. Fictional and flagged: is_demo = 1, tag 'demo', title prefix 'DEMO ·'.

The topic (a museum exhibit) is deliberately unrelated to any real project, so demo content
cannot be mistaken for research notes. Two pairs of records disagree on purpose, so the
conflict review has something to show.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import conflicts
from .db import transaction
from .repository import create_record, create_relationship, log_activity

BANNER = "> **Demonstration content.** Fictional sample data, not part of your knowledge base.\n\n"


def load_demo(conn: sqlite3.Connection) -> dict[str, Any]:
    existing = conn.execute("SELECT COUNT(*) FROM records WHERE is_demo = 1").fetchone()[0]
    if existing:
        return {"created": 0, "already_loaded": existing}

    def add(**fields: Any) -> str:
        fields["title"] = "DEMO · " + fields["title"]
        fields["content"] = BANNER + fields.get("content", "")
        fields["tags"] = ["demo", *fields.get("tags", [])]
        return create_record(conn, fields, is_demo=True, log=False)

    with transaction(conn):
        project = add(
            title="Harbor Lighthouse Exhibit",
            category="projects",
            kind="project",
            status="active",
            content="A small museum exhibit about a fictional harbor lighthouse.",
            metadata={"objectives": ["Open the exhibit in spring", "Digitize the keeper logs"]},
            tags=["exhibit"],
        )
        decision = add(
            title="Use an LED replica instead of the original lamp",
            category="projects",
            kind="decision",
            status="accepted",
            project_id=project,
            occurred_at="2026-10-08",
            content="**Decision:** display an LED replica.\n\n**Rationale:** the original lamp is fragile and"
            " cannot be powered safely.\n\n**Alternatives:** unlit original lamp; photograph only.",
            tags=["exhibit", "lighting"],
        )
        milestone = add(
            title="Draft exhibit floor plan",
            category="projects",
            kind="milestone",
            status="planned",
            project_id=project,
            occurred_at="2026-11-15",
            content="Floor plan with the lamp replica, the log cases and the audio corner.",
            tags=["exhibit"],
        )
        height_brochure = add(
            title="Lighthouse tower height (brochure)",
            category="facts",
            kind="fact",
            subject="Harbor lighthouse",
            predicate="tower height",
            value="31 m",
            confidence=0.6,
            source_ref="DEMO: museum brochure, page 2 (fictional)",
            content="The museum brochure gives the tower height as 31 m.",
            tags=["lighthouse"],
        )
        height_survey = add(
            title="Lighthouse tower height (survey)",
            category="facts",
            kind="fact",
            subject="Harbor lighthouse",
            predicate="tower height",
            value="34 m",
            confidence=0.8,
            source_ref="DEMO: 2019 survey sheet (fictional)",
            content="The survey sheet gives the tower height as 34 m.",
            tags=["lighthouse"],
        )
        first_lit = add(
            title="First lighting",
            category="facts",
            kind="fact",
            verified=True,
            confidence=0.9,
            source_ref="DEMO: harbor commission minutes (fictional)",
            occurred_at="1871-05-01",
            content="The harbor light was first lit in May 1871.",
            tags=["lighthouse", "history"],
        )
        logs_fact = add(
            title="Where the keeper logs are",
            category="facts",
            kind="fact",
            confidence=0.5,
            source_ref="DEMO: phone call with the archivist (fictional)",
            content="The original keeper logs are not stored in the town archive.",
            tags=["archive"],
        )
        visit = add(
            title="Archive visit",
            category="sessions",
            kind="session",
            occurred_at="2026-10-01",
            content="Visited the town archive.\n\n- Photographed two pages of the 1880 keeper log\n"
            "- Asked about the lamp's condition",
            metadata={"duration_minutes": 90, "activities": ["research", "photography"]},
            tags=["archive"],
        )
        sketch = add(
            title="Floor plan sketching",
            category="sessions",
            kind="session",
            occurred_at="2026-10-08",
            content="Sketched two layouts; chose the LED replica so the lamp can be lit.",
            metadata={"duration_minutes": 60, "activities": ["design"]},
            tags=["exhibit"],
        )
        add(
            title="Idea: audio guide narrated by a former keeper",
            category="inbox",
            kind="idea",
            content="Record a short audio guide; check whether a former keeper's family would take part.",
            tags=["exhibit"],
        )
        archive_note = add(
            title="Note from the archive desk",
            category="inbox",
            kind="observation",
            content="The original keeper logs are stored in the town archive.",
            tags=["archive"],
        )
        links = [
            (decision, sketch, "decided_in"),
            (milestone, decision, "depends_on"),
            (first_lit, project, "relates_to"),
            (sketch, visit, "follows"),
            (visit, archive_note, "relates_to"),
            (height_brochure, project, "relates_to"),
            (logs_fact, project, "relates_to"),
        ]
        for source, target, rel_type in links:
            create_relationship(conn, source, target, rel_type, origin="demo", log=False)
        log_activity(conn, "demo-loaded", None, "Demonstration data", "11 records")
    ids = [r[0] for r in conn.execute("SELECT id FROM records WHERE is_demo = 1")]
    found = conflicts.detect(conn, record_ids=ids)
    return {"created": len(ids), "conflicts": len(found["new_conflicts"]), "height_ids": [height_brochure, height_survey]}


def purge_demo(conn: sqlite3.Connection) -> dict[str, Any]:
    """Permanently remove demo records only (records flagged is_demo = 1)."""
    with transaction(conn):
        ids = [r[0] for r in conn.execute("SELECT id FROM records WHERE is_demo = 1")]
        for rid in ids:
            conn.execute("DELETE FROM records_fts WHERE record_id = ?", (rid,))
        conn.execute("DELETE FROM records WHERE is_demo = 1")
        if ids:
            log_activity(conn, "demo-removed", None, "Demonstration data", f"{len(ids)} records")
    return {"removed": len(ids)}
