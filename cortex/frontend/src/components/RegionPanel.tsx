import { useState } from "react";

import { api } from "../api";
import type { Nav } from "../App";
import { errorText, fmtDate, useAsync } from "../lib";
import type { BrainMap, Category, Cluster, RecordSummary, RegionId } from "../types";
import { CATEGORY_REGION, REGION_META } from "../types";
import { Badges, CategoryChip, RecordRow } from "./common";

export function RegionPanel({ region, map, rev, nav, onCluster }: { region: RegionId; map: BrainMap | null; rev: number; nav: Nav; onCluster: (c: Cluster) => void }) {
  const meta = REGION_META[region];
  const info = map?.regions.find((r) => r.region === region);
  return (
    <div className="region">
      <div className="mono tiny lime">{meta.anatomy} REGION</div>
      <h3 className="panel-title">
        {meta.label} <span className="mono muted">{info?.count ?? 0}</span>
      </h3>
      <p className="tiny muted">The brain region is a way to navigate this memory category, not a claim about what that part of the brain does.</p>
      {info && info.clusters.length > 0 && (
        <div className="clusters">
          {info.clusters.map((c) => (
            <button key={c.id} className="chip cluster-chip" onClick={() => onCluster(c)}>
              {c.label} <b>{c.count}</b>
            </button>
          ))}
        </div>
      )}
      {region === "prefrontal" && <Projects rev={rev} nav={nav} />}
      {region === "frontal" && <OpenConflicts rev={rev} nav={nav} />}
      {region === "temporal" && <Inbox rev={rev} nav={nav} />}
      {region === "hippocampus" && <Facts rev={rev} nav={nav} />}
      {region === "occipital" && <Sessions rev={rev} nav={nav} />}
    </div>
  );
}

export function ClusterPanel({ cluster, map, nav }: { cluster: Cluster; map: BrainMap | null; nav: Nav }) {
  const nodes = new Map((map?.nodes ?? []).map((n) => [n.id, n]));
  return (
    <div className="cluster">
      <div className="mono tiny lime">{REGION_META[cluster.region].anatomy} · CLUSTER</div>
      <h3 className="panel-title">
        {cluster.label} <span className="mono muted">{cluster.count}</span>
      </h3>
      {cluster.conflict_ids.length > 0 && (
        <p className="tiny">
          {cluster.conflict_ids.length} open conflict(s).{" "}
          <button className="link" onClick={() => nav.go({ kind: "conflicts" })}>
            Review
          </button>
        </p>
      )}
      <ul className="results">
        {cluster.record_ids.map((id) => {
          const n = nodes.get(id);
          return (
            <li key={id}>
              <button className="row-main" onClick={() => nav.openRecord(id)}>
                <span className="row-title">{n?.title ?? id}</span>
                {n && (
                  <span className="row-meta mono">
                    <CategoryChip category={n.category} /> {n.kind}
                    {n.is_demo && <span className="chip demo">DEMO</span>}
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function useCategory(category: Category, rev: number, extra: Parameters<typeof api.search>[0] = {}) {
  return useAsync(() => api.search({ category: [category], limit: 300, ...extra }), [category, rev, JSON.stringify(extra)]);
}

function Projects({ rev, nav }: { rev: number; nav: Nav }) {
  const projects = useAsync(() => api.projects(), [rev]);
  const decisions = useCategory("projects", rev, { kind: ["decision"], limit: 12 });
  return (
    <>
      <h4 className="section-title">Projects</h4>
      {projects.data?.length === 0 && (
        <p className="muted tiny">
          No projects yet.{" "}
          <button className="link" onClick={() => nav.go({ kind: "capture", mode: "project" })}>
            Create one
          </button>
        </p>
      )}
      {projects.data?.map((p) => (
        <div key={p.id} className="project-card">
          <button className="link row-title" onClick={() => nav.openRecord(p.id)}>
            {p.title}
          </button>
          <div className="row-meta mono">
            <span className="chip">{p.status ?? "active"}</span> <Badges r={p} />{" "}
            {Object.entries(p.record_counts)
              .map(([k, v]) => `${v} ${k}`)
              .join(" · ")}
          </div>
          {p.objectives.length > 0 && (
            <ul className="objectives">
              {p.objectives.map((o) => (
                <li key={o}>{o}</li>
              ))}
            </ul>
          )}
          {p.milestones.length > 0 && (
            <ul className="plain milestones mono tiny">
              {p.milestones.map((m) => (
                <li key={m.id}>
                  <span className="muted">{fmtDate(m.occurred_at)}</span>{" "}
                  <button className="link" onClick={() => nav.openRecord(m.id)}>
                    {m.title}
                  </button>{" "}
                  <span className="muted">{m.status}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
      <h4 className="section-title">Recent decisions</h4>
      <ul className="results">
        {decisions.data?.results.map((r) => <RecordRow key={r.id} r={r} nav={nav} />)}
      </ul>
      <button onClick={() => nav.go({ kind: "capture", mode: "decision" })}>+ Record a decision</button>
    </>
  );
}

function OpenConflicts({ rev, nav }: { rev: number; nav: Nav }) {
  const list = useAsync(() => api.conflicts("open"), [rev]);
  return (
    <>
      <h4 className="section-title">Open conflicts</h4>
      {list.data?.length === 0 && <p className="muted tiny">Nothing waiting for review.</p>}
      <ul className="results">
        {list.data?.map((c) => (
          <li key={c.id}>
            <button className="row-main" onClick={() => nav.go({ kind: "conflicts", id: c.id })}>
              <span className="row-title">
                {c.a.title} <span className="muted">vs</span> {c.b.title}
              </span>
              <span className="row-excerpt">{c.explanation}</span>
            </button>
          </li>
        ))}
      </ul>
      <button onClick={() => nav.go({ kind: "conflicts" })}>Open conflict review</button>
    </>
  );
}

function Inbox({ rev, nav }: { rev: number; nav: Nav }) {
  const items = useCategory("inbox", rev);
  const [busy, setBusy] = useState(false);
  const file = async (r: RecordSummary, category: Category) => {
    await api.updateRecord(r.id, { category, expected_updated_at: r.updated_at });
  };
  const withSuggestion = (items.data?.results ?? []).filter((r) => r.suggested_category && r.suggested_category !== "inbox");
  const run = async (fn: () => Promise<unknown>, msg: string) => {
    setBusy(true);
    try {
      await fn();
      nav.changed();
      nav.toast(msg);
    } catch (e) {
      nav.toast(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <h4 className="section-title">Waiting for triage</h4>
      <div className="toolbar">
        <button onClick={() => nav.go({ kind: "capture", mode: "note" })}>+ Quick note</button>
        {withSuggestion.length > 0 && (
          <button
            className="primary"
            disabled={busy}
            onClick={() =>
              window.confirm(`File ${withSuggestion.length} imported notes into their suggested categories?`) &&
              run(async () => {
                for (const r of withSuggestion) await file(r, r.suggested_category!);
              }, `Filed ${withSuggestion.length} notes.`)
            }
          >
            File all suggestions ({withSuggestion.length})
          </button>
        )}
      </div>
      {items.data?.results.length === 0 && <p className="muted tiny">Inbox is empty.</p>}
      <ul className="results">
        {items.data?.results.map((r) => (
          <RecordRow
            key={r.id}
            r={r}
            nav={nav}
            extra={
              <div className="row-actions">
                {r.suggested_category && r.suggested_category !== "inbox" && (
                  <button disabled={busy} onClick={() => run(() => file(r, r.suggested_category!), "Filed.")}>
                    → {REGION_META[CATEGORY_REGION[r.suggested_category]].label}
                  </button>
                )}
                <select value="" disabled={busy} onChange={(e) => e.target.value && run(() => file(r, e.target.value as Category), "Filed.")} aria-label="File into">
                  <option value="">file…</option>
                  {(["facts", "sessions", "projects"] as Category[]).map((c) => (
                    <option key={c} value={c}>
                      {REGION_META[CATEGORY_REGION[c]].label}
                    </option>
                  ))}
                </select>
              </div>
            }
          />
        ))}
      </ul>
    </>
  );
}

function Facts({ rev, nav }: { rev: number; nav: Nav }) {
  const [filter, setFilter] = useState<"all" | "verified" | "unverified">("all");
  const facts = useCategory("facts", rev, filter === "all" ? {} : { verified: filter === "verified" });
  return (
    <>
      <h4 className="section-title">Facts</h4>
      <div className="toolbar">
        <div className="seg">
          {(["all", "verified", "unverified"] as const).map((f) => (
            <button key={f} className={filter === f ? "active" : ""} onClick={() => setFilter(f)}>
              {f}
            </button>
          ))}
        </div>
        <button onClick={() => nav.go({ kind: "capture", mode: "fact" })}>+ Fact</button>
      </div>
      <ul className="results">
        {facts.data?.results.map((r) => <RecordRow key={r.id} r={r} nav={nav} />)}
      </ul>
    </>
  );
}

function Sessions({ rev, nav }: { rev: number; nav: Nav }) {
  const sessions = useCategory("sessions", rev);
  const groups = new Map<string, RecordSummary[]>();
  for (const s of sessions.data?.results ?? []) {
    const month = (s.occurred_at ?? s.created_at).slice(0, 7);
    if (!groups.has(month)) groups.set(month, []);
    groups.get(month)!.push(s);
  }
  return (
    <>
      <div className="toolbar">
        <button onClick={() => nav.go({ kind: "capture", mode: "session" })}>+ Session summary</button>
      </div>
      {[...groups.entries()].map(([month, items]) => (
        <div key={month} className="timeline">
          <div className="mono tiny lime">{month}</div>
          <ul className="plain">
            {items.map((s) => (
              <li key={s.id} className="timeline-item">
                <span className="mono muted">{(s.occurred_at ?? s.created_at).slice(5, 10)}</span>
                <button className="link" onClick={() => nav.openRecord(s.id)}>
                  {s.title}
                </button>
                <Badges r={s} />
                {s.excerpt && <div className="row-excerpt">{s.excerpt}</div>}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </>
  );
}
