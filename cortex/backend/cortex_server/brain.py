"""Data for the brain visualization: records grouped into clusters per region.

The brain is a navigation metaphor. A region shows one memory category; a cluster is a group
of records inside it (by kind, month, project, or conflict reason).
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

from .repository import REGIONS

REGION_ORDER = ("prefrontal", "frontal", "temporal", "hippocampus", "occipital")
REGION_INFO = {
    "prefrontal": {"category": "projects", "label": "Projects"},
    "frontal": {"category": "conflicts", "label": "Conflicts"},
    "temporal": {"category": "inbox", "label": "Inbox"},
    "hippocampus": {"category": "facts", "label": "Facts"},
    "occipital": {"category": "sessions", "label": "Sessions"},
}


_UNCOUNTABLE = {"literature", "research", "evidence", "information", "feedback"}


def _plural(kind: str) -> str:
    if kind in _UNCOUNTABLE or kind.endswith("s"):
        return kind
    if kind.endswith("y") and kind[-2:-1] not in "aeiou":
        return kind[:-1] + "ies"
    return f"{kind}s"


def brain_map(conn: sqlite3.Connection, include_demo: bool = True) -> dict[str, Any]:
    demo_clause = "" if include_demo else " AND r.is_demo = 0"
    rows = conn.execute(
        "SELECT r.id, r.title, r.category, r.kind, r.status, r.project_id, r.occurred_at, r.created_at,"
        "       r.verified, r.is_demo, p.title AS project_title"
        " FROM records r LEFT JOIN records p ON p.id = r.project_id"
        f" WHERE r.deleted_at IS NULL{demo_clause}"
        " ORDER BY COALESCE(r.occurred_at, r.created_at)"
    ).fetchall()

    clusters: dict[str, dict[str, Any]] = {}
    nodes = []

    def add(region: str, key: str, label: str, record_id: str) -> str:
        cid = f"{region}:{key}"
        cluster = clusters.setdefault(
            cid, {"id": cid, "region": region, "label": label, "count": 0, "record_ids": [], "conflict_ids": []}
        )
        cluster["count"] += 1
        cluster["record_ids"].append(record_id)
        return cid

    for r in rows:
        region = REGIONS[r["category"]]
        if r["category"] == "sessions":
            month = (r["occurred_at"] or r["created_at"])[:7]
            cid = add(region, month, month, r["id"])
        elif r["category"] == "projects":
            if r["kind"] == "project":
                cid = add(region, f"{r['id']}:project", r["title"], r["id"])
            else:
                owner = r["project_title"] or "Unassigned"
                cid = add(region, f"{r['project_id'] or 'none'}:{r['kind']}", f"{owner} · {_plural(r['kind'])}", r["id"])
        else:
            cid = add(region, r["kind"], _plural(r["kind"]), r["id"])
        nodes.append(
            {
                "id": r["id"],
                "title": r["title"],
                "category": r["category"],
                "kind": r["kind"],
                "region": region,
                "cluster_id": cid,
                "verified": bool(r["verified"]),
                "is_demo": bool(r["is_demo"]),
            }
        )

    active = {n["id"] for n in nodes}
    for c in conn.execute(
        "SELECT id, record_a_id, record_b_id, reason, status FROM conflicts WHERE status = 'open'"
    ):
        if c["record_a_id"] in active and c["record_b_id"] in active:
            cid = f"frontal:{c['reason']}"
            cluster = clusters.setdefault(
                cid,
                {"id": cid, "region": "frontal", "label": f"{c['reason']} conflicts", "count": 0,
                 "record_ids": [], "conflict_ids": []},
            )
            cluster["count"] += 1
            cluster["conflict_ids"].append(c["id"])
            for rid in (c["record_a_id"], c["record_b_id"]):
                if rid not in cluster["record_ids"]:
                    cluster["record_ids"].append(rid)

    edges = [
        {"id": e["id"], "source": e["source_id"], "target": e["target_id"], "rel_type": e["rel_type"]}
        for e in conn.execute("SELECT id, source_id, target_id, rel_type FROM relationships LIMIT 20000")
        if e["source_id"] in active and e["target_id"] in active
    ]

    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters.values():
        by_region[cluster["region"]].append(cluster)
    regions = []
    for region in REGION_ORDER:
        items = sorted(by_region[region], key=lambda c: (-c["count"], c["label"]))
        regions.append(
            {
                "region": region,
                **REGION_INFO[region],
                "count": sum(c["count"] for c in items),
                "clusters": items,
            }
        )
    return {"regions": regions, "nodes": nodes, "edges": edges}


def graph(
    conn: sqlite3.Connection, record_id: str | None = None, depth: int = 2, limit: int = 150
) -> dict[str, Any]:
    """Neighbourhood of a record (breadth-first), or the best-connected part of the whole graph."""
    adjacency: dict[str, list[tuple[str, sqlite3.Row]]] = defaultdict(list)
    edges_all = conn.execute(
        "SELECT rel.* FROM relationships rel JOIN records a ON a.id = rel.source_id JOIN records b ON b.id = rel.target_id"
        " WHERE a.deleted_at IS NULL AND b.deleted_at IS NULL"
    ).fetchall()
    for e in edges_all:
        adjacency[e["source_id"]].append((e["target_id"], e))
        adjacency[e["target_id"]].append((e["source_id"], e))

    selected: list[str] = []
    if record_id:
        seen = {record_id}
        frontier = [record_id]
        selected.append(record_id)
        for _ in range(max(depth, 0)):
            nxt = []
            for rid in frontier:
                for other, _edge in adjacency[rid]:
                    if other not in seen and len(selected) < limit:
                        seen.add(other)
                        selected.append(other)
                        nxt.append(other)
            frontier = nxt
    else:
        ranked = sorted(adjacency, key=lambda rid: -len(adjacency[rid]))
        selected = ranked[:limit]

    keep = set(selected)
    nodes = []
    for rid in selected:
        r = conn.execute(
            "SELECT id, title, category, kind, status, verified, is_demo, metadata FROM records WHERE id = ?", (rid,)
        ).fetchone()
        if r is None:
            continue
        nodes.append(
            {
                "id": r["id"],
                "title": r["title"],
                "category": r["category"],
                "kind": r["kind"],
                "status": r["status"],
                "verified": bool(r["verified"]),
                "is_demo": bool(r["is_demo"]),
                "superseded": "superseded_by" in json.loads(r["metadata"]),
                "degree": len(adjacency[rid]),
            }
        )
    edges = [
        {"id": e["id"], "source": e["source_id"], "target": e["target_id"], "rel_type": e["rel_type"], "note": e["note"]}
        for e in edges_all
        if e["source_id"] in keep and e["target_id"] in keep
    ]
    return {"root": record_id, "nodes": nodes, "edges": edges}
