import json

import pytest

from cortex_server import conflicts as cf
from cortex_server import repository as repo
from tests.conftest import make


def reasons(conn):
    return sorted(c["reason"] for c in cf.list_conflicts(conn, "open"))


def test_structured_facts_with_different_values(conn) -> None:
    make(conn, title="Height A", subject="Tower", predicate="height", value="31 m")
    make(conn, title="Height B", subject="tower", predicate="Height", value="34 m")
    make(conn, title="Height C", subject="Tower", predicate="height", value="31 M")  # same value as A
    cf.detect(conn)
    found = cf.list_conflicts(conn, "open")
    assert sorted(c["reason"] for c in found) == ["structured", "structured"]
    assert all("Tower" in c["excerpt_a"] or "tower" in c["excerpt_a"] for c in found)


def test_numeric_negation_and_antonym_statements(conn) -> None:
    make(conn, title="Rule 1", content="Perimeter columns are capped at 32 in. in the grouped design engine.")
    make(conn, title="Rule 2", content="Perimeter columns are capped at 36 in. in the grouped design engine.")
    make(conn, title="Slab 1", content="The no-slab branch retains slab mass and diaphragm constraints.")
    make(conn, title="Slab 2", content="The no-slab branch removes slab mass and diaphragm constraints.")
    make(conn, title="Vert 1", content="Vertical excitation is included in this phase of the study.")
    make(conn, title="Vert 2", content="Vertical excitation is not included in this phase of the study.")
    result = cf.detect(conn)
    assert len(result["new_conflicts"]) == 3
    assert reasons(conn) == ["antonym", "negation", "numeric"]
    numeric = next(c for c in cf.list_conflicts(conn) if c["reason"] == "numeric")
    assert "32 in" in numeric["explanation"] and "36 in" in numeric["explanation"]


def test_unrelated_or_differently_identified_statements_are_not_flagged(conn) -> None:
    make(conn, title="T01", content="Task T01 is planned for 60 min with the research team.")
    make(conn, title="T02", content="Task T02 is planned for 90 min with the research team.")
    make(conn, title="Same", content="The research team reviews every engineering packet.")
    make(conn, title="Same copy", content="The research team reviews every engineering packet.")
    make(conn, title="Other", content="Ground motions come from the NGA-West2 database.")
    cf.detect(conn)
    assert reasons(conn) == []


def test_citations_and_file_paths_name_sources_not_values(conn) -> None:
    make(conn, title="Earlier sources", content="Chou, Y.-T., Kuo, P.-C., Li, K.-Y., Chang, W.-T., & Chen, C.-S. (2025).")
    make(conn, title="Expanded sources", content="Kuo, P.-C., Chou, Y.-T., Li, K.-Y., Chang, W.-T., & Chen, C.-S. (2024).")
    make(conn, title="Bond slip", content="Original: `C:\\Users\\me\\ResearchProject\\2005-bond-slip-in-reinforced-concrete-elements.pdf`")
    make(conn, title="Deformations", content="Original: `C:\\Users\\me\\ResearchProject\\235489023-deformations-of-reinforced-concrete-members.pdf`")
    cf.detect(conn)
    assert reasons(conn) == []


def test_plain_years_in_prose_are_still_compared(conn) -> None:
    make(conn, title="A", content="The harbor light was first lit in 1871 by the commission.")
    make(conn, title="B", content="The harbor light was first lit in 1874 by the commission.")
    cf.detect(conn)
    assert reasons(conn) == ["numeric"]


def test_numbers_in_records_dated_on_different_days_are_progress_not_conflict(conn) -> None:
    make(conn, title="Day 1", category="sessions", occurred_at="2026-10-21",
         content="User-confirmed source tasks completed today: 0 of 4 with evidence.")
    make(conn, title="Day 2", category="sessions", occurred_at="2026-10-22",
         content="User-confirmed source tasks completed today: 3 of 4 with evidence.")
    cf.detect(conn)
    assert reasons(conn) == []


def test_template_lines_repeated_across_many_records_are_ignored(conn) -> None:
    for i in range(5):
        make(conn, title=f"Exp {i}", content=f"Runtime and resource budget for this experiment is {i + 1} hours total.")
    cf.detect(conn)
    assert reasons(conn) == []


def test_demo_and_real_records_are_never_compared(conn) -> None:
    make(conn, title="Real", subject="Tower", predicate="height", value="31 m")
    repo.create_record(conn, {"title": "Demo", "category": "facts", "subject": "Tower", "predicate": "height",
                              "value": "99 m"}, is_demo=True)
    cf.detect(conn)
    assert reasons(conn) == []


def test_detection_never_modifies_records(conn) -> None:
    a = make(conn, title="A", subject="S", predicate="p", value="1")
    b = make(conn, title="B", subject="S", predicate="p", value="2")
    before = [dict(r) for r in conn.execute("SELECT * FROM records ORDER BY id")]
    cf.detect(conn)
    assert [dict(r) for r in conn.execute("SELECT * FROM records ORDER BY id")] == before
    assert {a, b} == {c["a"]["id"] for c in cf.list_conflicts(conn)} | {c["b"]["id"] for c in cf.list_conflicts(conn)}


def test_prefer_resolution_links_and_annotates_without_deleting_or_rewriting(conn) -> None:
    a = make(conn, title="A", subject="S", predicate="p", value="1", content="original A")
    b = make(conn, title="B", subject="S", predicate="p", value="2", content="original B")
    cf.detect(conn)
    conflict = cf.list_conflicts(conn, "open")[0]
    keep = conflict["a"]["id"]
    other = conflict["b"]["id"]
    cf.resolve_conflict(conn, conflict["id"], "prefer_a", "survey is newer")
    resolved = cf.get_conflict(conn, conflict["id"])
    assert resolved["status"] == "resolved" and resolved["resolution"] == "prefer_a"
    other_rec = repo.get_record(conn, other)
    assert other_rec["deleted"] is False
    assert other_rec["content"] in ("original A", "original B")
    assert other_rec["metadata"]["superseded_by"] == keep
    assert [r["rel_type"] for r in repo.relationships_for(conn, keep)["outgoing"]] == ["supersedes"]
    assert repo.stats(conn)["open_conflicts"] == 0
    with pytest.raises(repo.InvalidInput):
        cf.resolve_conflict(conn, conflict["id"], "keep_both")

    cf.reopen_conflict(conn, conflict["id"])
    assert cf.get_conflict(conn, conflict["id"])["status"] == "open"
    assert "superseded_by" not in repo.get_record(conn, other)["metadata"]
    assert repo.relationships_for(conn, keep)["outgoing"] == []
    assert {a, b} == {keep, other}


def test_dismissed_candidates_are_not_raised_again(conn) -> None:
    make(conn, title="A", subject="S", predicate="p", value="1")
    make(conn, title="B", subject="S", predicate="p", value="2")
    cf.detect(conn)
    conflict = cf.list_conflicts(conn)[0]
    cf.resolve_conflict(conn, conflict["id"], "not_a_conflict", "different units")
    assert cf.detect(conn)["new_conflicts"] == []
    assert cf.list_conflicts(conn)[0]["status"] == "dismissed"


def test_superseded_pairs_are_skipped_by_later_scans(conn) -> None:
    a = make(conn, title="A", subject="S", predicate="p", value="1")
    b = make(conn, title="B", subject="S", predicate="p", value="2")
    repo.create_relationship(conn, a, b, "supersedes")
    assert cf.detect(conn)["new_conflicts"] == []


def test_manual_flag(conn) -> None:
    a = make(conn, title="A")
    b = make(conn, title="B")
    cid = cf.flag_conflict(conn, a, b, "claim one", "claim two", "They disagree about scope.")
    c = cf.get_conflict(conn, cid)
    assert c["reason"] == "manual" and c["record_a"]["title"] == "A"
    with pytest.raises(repo.InvalidInput):
        cf.flag_conflict(conn, a, a, "", "", "")


def test_resolution_is_versioned(conn) -> None:
    make(conn, title="A", subject="S", predicate="p", value="1")
    make(conn, title="B", subject="S", predicate="p", value="2")
    cf.detect(conn)
    conflict = cf.list_conflicts(conn)[0]
    cf.resolve_conflict(conn, conflict["id"], "prefer_b")
    versions = repo.record_versions(conn, conflict["a"]["id"])
    assert versions and "resolved in favour" in versions[0]["reason"]
    assert json.loads(versions[0]["snapshot"]["metadata"]) == {}
