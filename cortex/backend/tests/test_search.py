import pytest

from cortex_server import repository as repo
from cortex_server.search import SNIPPET_CLOSE, SNIPPET_OPEN, Filters, build_match_query, search, similar
from tests.conftest import make


@pytest.fixture
def corpus(conn):
    p = make(conn, title="SeisFrame V6", category="projects", kind="project")
    ids = {
        "imk": make(conn, title="IMK deterioration modes", content="Basic strength and post-capping strength deterioration.",
                    tags=["imk"], project_id=p, occurred_at="2026-10-22"),
        "theta": make(conn, title="Ultimate rotation", content="theta_u is measured from the origin.",
                      tags=["imk", "rotation"], occurred_at="2026-10-21", verified=True),
        "slab": make(conn, title="No-slab branch", content="Removes slab elements; retains slab mass.",
                     category="projects", kind="decision", project_id=p, occurred_at="2026-10-20"),
        "session": make(conn, title="Oral exam session", content="Discussed IMK energy normalization lambda.",
                        category="sessions", kind="session", occurred_at="2026-10-22"),
    }
    ids["project"] = p
    return ids


def titles(result):
    return [r["title"] for r in result["results"]]


def test_full_text_ranks_title_matches_first(conn, corpus) -> None:
    result = search(conn, "deterioration", Filters())
    assert result["mode"] == "fulltext"
    assert titles(result)[0] == "IMK deterioration modes"


def test_prefix_phrase_and_exclusion(conn, corpus) -> None:
    assert "IMK deterioration modes" in titles(search(conn, "deterior", Filters()))
    assert titles(search(conn, '"post-capping strength"', Filters())) == ["IMK deterioration modes"]
    assert "Oral exam session" not in titles(search(conn, "imk -energy", Filters()))


@pytest.mark.parametrize("query", ['imk OR', 'NEAR(a b', '"unbalanced', '*', 'title:imk', 'a AND', '(((', "'; DROP TABLE records; --"])
def test_fts_syntax_from_users_cannot_break_queries(conn, corpus, query) -> None:
    result = search(conn, query, Filters())
    assert "error" not in result
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 5


def test_build_match_query_quotes_every_token() -> None:
    assert build_match_query("imk OR theta") == '"imk"* AND "OR"* AND "theta"*'
    assert build_match_query('"exact phrase" -skip') == '"exact phrase" NOT "skip"*'
    assert build_match_query("   ") is None
    assert build_match_query("-only -negatives") is None


def test_filters_category_tag_date_project_verified(conn, corpus) -> None:
    assert titles(search(conn, "", Filters(categories=["sessions"]))) == ["Oral exam session"]
    assert set(titles(search(conn, "", Filters(tags=["imk"])))) == {"IMK deterioration modes", "Ultimate rotation"}
    assert set(titles(search(conn, "", Filters(tags=["imk", "rotation"])))) == {"Ultimate rotation"}
    dated = search(conn, "", Filters(date_from="2026-10-21", date_to="2026-10-21"))
    assert titles(dated) == ["Ultimate rotation"]
    in_project = set(titles(search(conn, "", Filters(project_id=corpus["project"]))))
    assert in_project == {"SeisFrame V6", "IMK deterioration modes", "No-slab branch"}
    assert titles(search(conn, "", Filters(verified=True))) == ["Ultimate rotation"]


def test_trashed_and_demo_records_are_filtered(conn, corpus) -> None:
    repo.trash_record(conn, corpus["theta"])
    assert "Ultimate rotation" not in titles(search(conn, "theta", Filters()))
    repo.create_record(conn, {"title": "Demo IMK", "category": "facts"}, is_demo=True)
    assert "Demo IMK" in titles(search(conn, "imk", Filters()))
    assert "Demo IMK" not in titles(search(conn, "imk", Filters(include_demo=False)))


def test_snippet_marks_matches(conn, corpus) -> None:
    result = search(conn, "origin", Filters())
    snippet = result["results"][0]["snippet"]
    assert f"{SNIPPET_OPEN}origin{SNIPPET_CLOSE}" in snippet


def test_edits_are_searchable_immediately(conn, corpus) -> None:
    repo.update_record(conn, corpus["slab"], {"content": "Flange participation is removed."})
    assert titles(search(conn, "flange", Filters())) == ["No-slab branch"]
    assert search(conn, "retains", Filters())["total"] == 0
    repo.update_record(conn, corpus["slab"], {"tags": ["branching"]})
    assert titles(search(conn, "branching", Filters())) == ["No-slab branch"]


def test_similarity_is_lexical_and_labelled(conn, corpus) -> None:
    result = similar(conn, record_id=corpus["imk"])
    assert result["method"] == "tfidf-cosine"
    assert corpus["imk"] not in [r["id"] for r in result["results"]]
    assert result["results"][0]["title"] in {"Oral exam session", "Ultimate rotation"}
    by_text = similar(conn, text="slab mass elements")
    assert by_text["results"][0]["title"] == "No-slab branch"
