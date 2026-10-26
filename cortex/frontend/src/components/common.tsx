import { useEffect, useState } from "react";

import { api } from "../api";
import type { Nav } from "../App";
import type { Category, RecordSummary } from "../types";
import { CATEGORY_REGION, REGION_META } from "../types";

export function CategoryChip({ category }: { category: Category }) {
  const region = CATEGORY_REGION[category];
  return (
    <span className={`chip cat-${category}`} title={`${REGION_META[region].anatomy} region`}>
      {REGION_META[region].label}
    </span>
  );
}

export function Badges({ r }: { r: Pick<RecordSummary, "verified" | "is_demo" | "deleted" | "superseded_by" | "category"> }) {
  return (
    <>
      {r.is_demo && <span className="chip demo">DEMO</span>}
      {r.category === "facts" && (r.verified ? <span className="chip verified">verified</span> : <span className="chip">unverified</span>)}
      {r.superseded_by && <span className="chip warn">superseded</span>}
      {r.deleted && <span className="chip danger">in trash</span>}
    </>
  );
}

export function RecordRow({ r, nav, extra }: { r: RecordSummary; nav: Nav; extra?: React.ReactNode }) {
  return (
    <li className="record-row">
      <button className="row-main" onClick={() => nav.openRecord(r.id)}>
        <span className="row-title">{r.title}</span>
        <span className="row-meta mono">
          <CategoryChip category={r.category} /> {r.kind}
          {r.status ? ` · ${r.status}` : ""}
          {r.occurred_at ? ` · ${r.occurred_at.slice(0, 10)}` : ""} <Badges r={r} />
        </span>
        {r.excerpt && <span className="row-excerpt">{r.excerpt}</span>}
      </button>
      {extra}
    </li>
  );
}

/** Type-ahead record search used for linking records. */
export function RecordPicker({ onPick, exclude = [], placeholder = "Find a record to link…" }: { onPick: (r: RecordSummary) => void; exclude?: string[]; placeholder?: string }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<RecordSummary[]>([]);
  const excludeKey = exclude.join(",");
  useEffect(() => {
    if (q.trim().length < 2) {
      setResults([]);
      return;
    }
    const skip = new Set(excludeKey.split(","));
    const handle = window.setTimeout(() => {
      api
        .search({ q, limit: 8, include_documents: false })
        .then((r) => setResults(r.results.filter((x) => !skip.has(x.id))))
        .catch(() => setResults([]));
    }, 180);
    return () => window.clearTimeout(handle);
  }, [q, excludeKey]);
  return (
    <div className="picker">
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={placeholder} />
      {results.length > 0 && (
        <ul className="picker-results">
          {results.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                onClick={() => {
                  onPick(r);
                  setQ("");
                  setResults([]);
                }}
              >
                <span>{r.title}</span> <CategoryChip category={r.category} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function useProjects(rev = 0) {
  const [projects, setProjects] = useState<{ id: string; title: string }[]>([]);
  useEffect(() => {
    api
      .projects()
      .then((p) => setProjects(p.map(({ id, title }) => ({ id, title }))))
      .catch(() => setProjects([]));
  }, [rev]);
  return projects;
}

export function ProjectSelect({ value, onChange, rev }: { value: string | null; onChange: (id: string | null) => void; rev?: number }) {
  const projects = useProjects(rev);
  return (
    <select value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
      <option value="">— no project —</option>
      {projects.map((p) => (
        <option key={p.id} value={p.id}>
          {p.title}
        </option>
      ))}
    </select>
  );
}

export function TagInput({ value, onChange }: { value: string[]; onChange: (tags: string[]) => void }) {
  const [draft, setDraft] = useState("");
  const commit = () => {
    const parts = draft.split(/[,\s]+/).map((t) => t.trim().replace(/^#/, "").toLowerCase()).filter(Boolean);
    if (parts.length) onChange([...new Set([...value, ...parts])]);
    setDraft("");
  };
  return (
    <div className="tag-input">
      {value.map((t) => (
        <span key={t} className="chip tag">
          #{t}
          <button type="button" aria-label={`remove ${t}`} onClick={() => onChange(value.filter((x) => x !== t))}>
            ×
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            commit();
          }
        }}
        onBlur={commit}
        placeholder="add tag"
      />
    </div>
  );
}
