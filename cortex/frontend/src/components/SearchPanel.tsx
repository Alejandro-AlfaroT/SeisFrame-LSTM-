import { useEffect, useState } from "react";

import { api, type SearchParams } from "../api";
import type { Nav } from "../App";
import { errorText, Snippet, useAsync } from "../lib";
import type { Category, SearchResult } from "../types";
import { CATEGORY_REGION, REGION_META } from "../types";
import { Badges, CategoryChip, ProjectSelect } from "./common";

const CATEGORIES: Category[] = ["facts", "sessions", "inbox", "projects"];

export default function SearchPanel({ initialQuery, nav, onResults }: { initialQuery: string; nav: Nav; onResults: (ids: string[]) => void }) {
  const [q, setQ] = useState(initialQuery);
  const [mode, setMode] = useState<"fulltext" | "similar">("fulltext");
  const [categories, setCategories] = useState<Category[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [verifiedOnly, setVerifiedOnly] = useState(false);
  const [includeDemo, setIncludeDemo] = useState(true);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"notes" | "pdfs">("notes");
  const allTags = useAsync(() => api.tags(), []);

  useEffect(() => setQ(initialQuery), [initialQuery]);

  useEffect(() => {
    const params: SearchParams = {
      q,
      mode,
      category: categories,
      tag: tags,
      project_id: projectId ?? undefined,
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
      verified: verifiedOnly ? true : undefined,
      include_demo: includeDemo,
      limit: 100,
    };
    const handle = window.setTimeout(() => {
      api
        .search(params)
        .then((r) => {
          setResult(r);
          setError(r.error ?? r.documents?.error ?? null);
          onResults([...r.results.map((x) => x.id), ...(r.documents?.results.map((d) => d.id) ?? [])]);
        })
        .catch((e) => setError(errorText(e)));
    }, 160);
    return () => window.clearTimeout(handle);
  }, [q, mode, categories, tags, projectId, dateFrom, dateTo, verifiedOnly, includeDemo, onResults]);

  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  // PDF passages exist only for a full-text query; when no note matches but pages do, show those.
  const docs = result?.mode === "fulltext" && result.documents && result.documents.total_documents > 0 ? result.documents : null;
  const shown = docs && (view === "pdfs" || result!.total === 0) ? "pdfs" : "notes";

  return (
    <div className="search">
      <input className="search-input" autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder={mode === "fulltext" ? 'words, "exact phrase", -exclude' : "describe what you are looking for"} />
      <div className="seg">
        <button className={mode === "fulltext" ? "active" : ""} onClick={() => setMode("fulltext")}>
          Full text <span className="mono tiny">FTS5</span>
        </button>
        <button className={mode === "similar" ? "active" : ""} onClick={() => setMode("similar")}>
          Similar <span className="mono tiny">tf-idf</span>
        </button>
      </div>
      <div className="filters">
        <div className="filter-row">
          {CATEGORIES.map((c) => (
            <label key={c} className={`check${categories.includes(c) ? " on" : ""}`}>
              <input type="checkbox" checked={categories.includes(c)} onChange={() => setCategories(toggle(categories, c))} />
              {REGION_META[CATEGORY_REGION[c]].label}
            </label>
          ))}
        </div>
        <div className="filter-row">
          <ProjectSelect value={projectId} onChange={setProjectId} />
          <select value="" onChange={(e) => e.target.value && setTags(toggle(tags, e.target.value))} aria-label="Filter by tag">
            <option value="">+ tag filter</option>
            {(allTags.data ?? []).map((t) => (
              <option key={t.tag} value={t.tag}>
                #{t.tag} ({t.count})
              </option>
            ))}
          </select>
        </div>
        <div className="filter-row mono tiny">
          <label>
            from <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
          </label>
          <label>
            to <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
          </label>
        </div>
        <div className="filter-row mono tiny">
          <label className="check">
            <input type="checkbox" checked={verifiedOnly} onChange={(e) => setVerifiedOnly(e.target.checked)} /> verified only
          </label>
          <label className="check">
            <input type="checkbox" checked={includeDemo} onChange={(e) => setIncludeDemo(e.target.checked)} /> include demo
          </label>
        </div>
        {tags.length > 0 && (
          <div className="filter-row">
            {tags.map((t) => (
              <button key={t} className="chip tag" onClick={() => setTags(tags.filter((x) => x !== t))}>
                #{t} ×
              </button>
            ))}
          </div>
        )}
      </div>
      {error && <p className="notice warn">{error}</p>}
      {docs && (
        <div className="seg">
          <button className={shown === "notes" ? "active" : ""} onClick={() => setView("notes")}>
            Notes <span className="mono tiny">{result!.total}</span>
          </button>
          <button className={shown === "pdfs" ? "active" : ""} onClick={() => setView("pdfs")}>
            In PDFs{" "}
            <span className="mono tiny">
              {docs.total_pages}
              {docs.capped ? "+" : ""} page{docs.total_pages === 1 ? "" : "s"} · {docs.total_documents} file{docs.total_documents === 1 ? "" : "s"}
            </span>
          </button>
        </div>
      )}
      {result && shown === "notes" && (
        <p className="mono tiny muted">
          {result.total} result{result.total === 1 ? "" : "s"} · {result.mode === "browse" ? "newest first" : result.mode === "similar" ? "lexical similarity (not neural embeddings)" : "ranked by bm25"} · lit in the brain
        </p>
      )}
      {docs && shown === "pdfs" && (
        <>
          <p className="mono tiny muted">
            pages ranked by bm25 · best 3 per file
            {docs.total_documents > docs.results.length ? ` · top ${docs.results.length} of ${docs.total_documents} files` : ""} · lit in the brain
          </p>
          <ul className="results">
            {docs.results.map((d) => (
              <li key={d.id}>
                <button className="row-main" onClick={() => nav.openRecord(d.id, { page: d.passages[0]?.page ?? 1, q })}>
                  <span className="row-title">{d.title}</span>
                  <span className="row-meta mono">
                    <CategoryChip category={d.category} /> pdf · {d.pages_matched} of {d.page_count} pages match <Badges r={d} />
                  </span>
                </button>
                <ul className="passages">
                  {d.passages.map((p) => (
                    <li key={p.page}>
                      <button className="passage" onClick={() => nav.openRecord(d.id, { page: p.page, q })}>
                        <span className="mono page-no">p. {p.page}</span>
                        <span className="row-excerpt">
                          <Snippet text={p.snippet} />
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </>
      )}
      <ul className="results">
        {shown === "notes" && result?.results.map((r) => (
          <li key={r.id}>
            <button className="row-main" onClick={() => nav.openRecord(r.id)}>
              <span className="row-title">{r.title}</span>
              <span className="row-meta mono">
                <CategoryChip category={r.category} /> {r.kind}
                {r.occurred_at ? ` · ${r.occurred_at.slice(0, 10)}` : ""}
                {r.score != null && result.mode === "similar" ? ` · ${r.score.toFixed(2)}` : ""} <Badges r={r} />
              </span>
              <span className="row-excerpt">{r.snippet ? <Snippet text={r.snippet} /> : r.excerpt}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
