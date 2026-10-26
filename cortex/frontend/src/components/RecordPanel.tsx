import { Fragment, useState } from "react";

import { api, type RecordPatch } from "../api";
import type { Nav } from "../App";
import { errorText, fmtDate, Snippet, useAsync } from "../lib";
import type { Category, DocumentInfo, MemoryRecord, RelationshipView } from "../types";
import { CATEGORY_REGION, REGION_META } from "../types";
import { Badges, CategoryChip, ProjectSelect, RecordPicker, TagInput } from "./common";
import Markdown from "./Markdown";

const CATEGORIES: Category[] = ["inbox", "facts", "sessions", "projects"];

type Draft = Required<Pick<RecordPatch, "title" | "content" | "kind" | "tags">> & Omit<RecordPatch, "title" | "content" | "kind" | "tags">;

function toDraft(r: MemoryRecord): Draft {
  return {
    title: r.title,
    content: r.content,
    kind: r.kind,
    tags: r.tags,
    status: r.status,
    project_id: r.project_id,
    source_ref: r.source_ref,
    confidence: r.confidence,
    subject: r.subject,
    predicate: r.predicate,
    value: r.value,
    occurred_at: r.occurred_at,
  };
}

export default function RecordPanel({ id, rev, nav, page, q }: { id: string; rev: number; nav: Nav; page?: number; q?: string }) {
  const rec = useAsync(() => api.record(id), [id, rev]);
  const relTypes = useAsync(() => api.relationshipTypes(), []);
  const similar = useAsync(() => api.similar(id), [id, rev]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showVersions, setShowVersions] = useState(false);
  const [linkType, setLinkType] = useState("relates_to");
  const [flagging, setFlagging] = useState(false);

  if (rec.error) return <p className="warn pad">{rec.error}</p>;
  const r = rec.data;
  if (!r) return <p className="muted pad mono">loading…</p>;

  const act = async (fn: () => Promise<unknown>, done?: string) => {
    setBusy(true);
    setErr(null);
    try {
      await fn();
      nav.changed();
      if (done) nav.toast(done);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  const patch = (p: RecordPatch, done?: string) => act(() => api.updateRecord(r.id, { ...p, expected_updated_at: r.updated_at }), done);

  const save = () =>
    act(async () => {
      await api.updateRecord(r.id, { ...draft!, expected_updated_at: r.updated_at });
      setDraft(null);
    }, "Saved. The previous version is in the history.");

  const suggested = (r.metadata.suggested_category as Category | undefined) ?? null;
  const importMeta = r.metadata.import as { path?: string; sha256?: string; imported_at?: string; unresolved_links?: string[]; root?: string } | undefined;
  const frontmatter = r.metadata.frontmatter as Record<string, unknown> | undefined;
  const supersededBy = r.metadata.superseded_by as string | undefined;
  const sourceIsUrl = r.source_ref ? /^https?:\/\//i.test(r.source_ref) : false;

  return (
    <article className="record">
      <div className="record-chips">
        <CategoryChip category={r.category} />
        <span className="chip">{r.kind}</span>
        {r.status && <span className="chip">{r.status}</span>}
        <Badges r={{ ...r, superseded_by: supersededBy ?? null }} />
      </div>

      {draft ? (
        <input className="title-input" value={draft.title} onChange={(e) => setDraft({ ...draft, title: e.target.value })} />
      ) : (
        <h2 className="record-title">{r.title}</h2>
      )}

      {supersededBy && (
        <p className="notice warn">
          Superseded after conflict review by{" "}
          <button className="link" onClick={() => nav.openRecord(supersededBy)}>
            another record
          </button>
          . This record is kept unchanged.
        </p>
      )}
      {r.deleted && (
        <p className="notice danger">
          In the trash since {fmtDate(r.deleted_at, true)}.{" "}
          <button className="link" disabled={busy} onClick={() => act(() => api.restoreRecord(r.id), "Restored.")}>
            Restore
          </button>
        </p>
      )}

      <dl className="meta-grid mono">
        <dt>region</dt>
        <dd>{REGION_META[CATEGORY_REGION[r.category]].anatomy.toLowerCase()}</dd>
        <dt>created</dt>
        <dd>{fmtDate(r.created_at, true)}</dd>
        <dt>updated</dt>
        <dd>{fmtDate(r.updated_at, true)}</dd>
        <dt>date</dt>
        <dd>{draft ? <input type="date" value={draft.occurred_at?.slice(0, 10) ?? ""} onChange={(e) => setDraft({ ...draft, occurred_at: e.target.value || null })} /> : fmtDate(r.occurred_at)}</dd>
        <dt>project</dt>
        <dd>
          {draft ? (
            <ProjectSelect value={draft.project_id ?? null} onChange={(v) => setDraft({ ...draft, project_id: v })} />
          ) : r.project ? (
            <button className="link" onClick={() => nav.openRecord(r.project!.id)}>
              {r.project.title}
            </button>
          ) : (
            "—"
          )}
        </dd>
        <dt>source</dt>
        <dd className="break">
          {draft ? (
            <input value={draft.source_ref ?? ""} onChange={(e) => setDraft({ ...draft, source_ref: e.target.value || null })} placeholder="URL, DOI, file, page…" />
          ) : sourceIsUrl ? (
            <a href={r.source_ref!} target="_blank" rel="noopener noreferrer nofollow">
              {r.source_ref}
            </a>
          ) : (
            (r.source_ref ?? "—")
          )}
        </dd>
        {(r.category === "facts" || r.confidence !== null) && (
          <>
            <dt>confidence</dt>
            <dd>
              {draft ? (
                <input type="number" min={0} max={1} step={0.05} value={draft.confidence ?? ""} onChange={(e) => setDraft({ ...draft, confidence: e.target.value === "" ? null : Number(e.target.value) })} />
              ) : r.confidence !== null ? (
                r.confidence.toFixed(2)
              ) : (
                "—"
              )}
            </dd>
          </>
        )}
        {r.category === "facts" && (
          <>
            <dt>verified</dt>
            <dd>{r.verified ? `yes · ${fmtDate(r.verified_at, true)}` : "no"}</dd>
          </>
        )}
        {draft && (
          <>
            <dt>kind</dt>
            <dd>
              <input value={draft.kind} onChange={(e) => setDraft({ ...draft, kind: e.target.value })} />
            </dd>
            <dt>status</dt>
            <dd>
              <input value={draft.status ?? ""} onChange={(e) => setDraft({ ...draft, status: e.target.value || null })} />
            </dd>
          </>
        )}
      </dl>

      {(draft || r.subject) && (
        <div className="triple mono">
          {draft ? (
            <>
              <input placeholder="subject" value={draft.subject ?? ""} onChange={(e) => setDraft({ ...draft, subject: e.target.value || null })} />
              <input placeholder="predicate" value={draft.predicate ?? ""} onChange={(e) => setDraft({ ...draft, predicate: e.target.value || null })} />
              <input placeholder="value" value={draft.value ?? ""} onChange={(e) => setDraft({ ...draft, value: e.target.value || null })} />
            </>
          ) : (
            <span>
              {r.subject} · {r.predicate} = <b className="lime">{r.value}</b>
            </span>
          )}
        </div>
      )}

      <div className="tags-line">{draft ? <TagInput value={draft.tags} onChange={(tags) => setDraft({ ...draft, tags })} /> : r.tags.map((t) => <span key={t} className="chip tag">#{t}</span>)}</div>

      {err && <p className="notice warn">{err}</p>}

      <div className="toolbar">
        {draft ? (
          <>
            <button className="primary" disabled={busy || !draft.title.trim()} onClick={save}>
              Save
            </button>
            <button onClick={() => setDraft(null)} disabled={busy}>
              Cancel
            </button>
          </>
        ) : (
          !r.deleted && (
            <>
              <button onClick={() => setDraft(toDraft(r))}>Edit</button>
              {r.category === "facts" && (
                <button className={r.verified ? "" : "primary"} disabled={busy} onClick={() => patch({ verified: !r.verified }, r.verified ? "Marked unverified." : "Marked verified.")}>
                  {r.verified ? "Unverify" : "Mark verified"}
                </button>
              )}
              {r.category === "inbox" && suggested && suggested !== "inbox" && (
                <button className="primary" disabled={busy} onClick={() => patch({ category: suggested }, `Filed under ${REGION_META[CATEGORY_REGION[suggested]].label}.`)}>
                  File → {REGION_META[CATEGORY_REGION[suggested]].label}
                </button>
              )}
              <select value="" onChange={(e) => e.target.value && patch({ category: e.target.value as Category }, "Moved.")} aria-label="Move to category">
                <option value="">Move to…</option>
                {CATEGORIES.filter((c) => c !== r.category).map((c) => (
                  <option key={c} value={c}>
                    {REGION_META[CATEGORY_REGION[c]].label}
                  </option>
                ))}
              </select>
              <button onClick={() => nav.go({ kind: "graph", id: r.id })}>Graph</button>
              <button onClick={() => setFlagging((f) => !f)}>Flag conflict</button>
              <button
                className="danger"
                disabled={busy}
                onClick={() => window.confirm("Move this record to the trash? You can restore it from Data → Trash.") && act(() => api.trashRecord(r.id), "Moved to trash.")}
              >
                Trash
              </button>
            </>
          )
        )}
      </div>

      {flagging && (
        <div className="subpanel">
          <p className="tiny muted">Pick the record that contradicts this one. You will review it in the conflict queue.</p>
          <RecordPicker
            exclude={[r.id]}
            onPick={(other) => {
              const explanation = window.prompt("What do they disagree about?", "") ?? "";
              act(async () => {
                const c = await api.flagConflict({ record_a_id: r.id, record_b_id: other.id, excerpt_a: "", excerpt_b: "", explanation });
                setFlagging(false);
                nav.go({ kind: "conflicts", id: c.id });
              });
            }}
          />
        </div>
      )}

      <section className="content">
        {draft ? (
          <textarea className="content-input" value={draft.content} onChange={(e) => setDraft({ ...draft, content: e.target.value })} rows={18} />
        ) : r.content.trim() ? (
          <Markdown text={r.content} onOpenRef={nav.openRef} />
        ) : (
          <p className="muted">No content.</p>
        )}
      </section>

      {r.document && <PdfPages id={r.id} doc={r.document} initialPage={page} initialQuery={q} />}

      <section>
        <h4 className="section-title">Relationships</h4>
        <RelList title="outgoing" items={r.relationships.outgoing} types={relTypes.data ?? []} nav={nav} busy={busy} act={act} arrow="→" />
        <RelList title="incoming" items={r.relationships.incoming} types={relTypes.data ?? []} nav={nav} busy={busy} act={act} arrow="←" />
        {!r.deleted && (
          <div className="add-link">
            <select value={linkType} onChange={(e) => setLinkType(e.target.value)} aria-label="Relationship type">
              {(relTypes.data ?? ["relates_to"]).map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <RecordPicker exclude={[r.id]} onPick={(other) => act(() => api.createRelationship(r.id, other.id, linkType), `Linked: ${linkType} → ${other.title}`)} />
          </div>
        )}
      </section>

      {r.conflicts.length > 0 && (
        <section>
          <h4 className="section-title">Conflicts</h4>
          <ul className="plain">
            {r.conflicts.map((c) => (
              <li key={c.id}>
                <button className="link" onClick={() => nav.go({ kind: "conflicts", id: c.id })}>
                  {c.reason} · {c.status}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h4 className="section-title">
          Similar records <span className="muted tiny mono">tf-idf · lexical</span>
        </h4>
        <ul className="plain">
          {(similar.data?.results ?? []).map((s) => (
            <li key={s.id} className="similar">
              <span className="mono muted">{(s.score ?? 0).toFixed(2)}</span>
              <button className="link" onClick={() => nav.openRecord(s.id)}>
                {s.title}
              </button>
            </li>
          ))}
          {similar.data && similar.data.results.length === 0 && <li className="muted tiny">None.</li>}
        </ul>
      </section>

      <section>
        <h4 className="section-title">
          <button className="link" onClick={() => setShowVersions((v) => !v)}>
            {showVersions ? "▾" : "▸"} Version history ({r.version_count})
          </button>
        </h4>
        {showVersions && <Versions id={r.id} rev={rev} act={act} busy={busy} />}
      </section>

      {(importMeta || frontmatter) && (
        <details className="provenance">
          <summary className="section-title">Provenance</summary>
          <dl className="meta-grid mono">
            {importMeta?.root && (
              <>
                <dt>folder</dt>
                <dd className="break">{importMeta.root}</dd>
              </>
            )}
            <dt>file</dt>
            <dd className="break">{importMeta?.path ?? "—"}</dd>
            <dt>imported</dt>
            <dd>{fmtDate(importMeta?.imported_at, true)}</dd>
            <dt>sha256</dt>
            <dd className="break">{importMeta?.sha256?.slice(0, 16)}…</dd>
            {importMeta?.unresolved_links?.length ? (
              <>
                <dt>unresolved</dt>
                <dd className="break">{importMeta.unresolved_links.join(", ")}</dd>
              </>
            ) : null}
          </dl>
          {frontmatter && Object.keys(frontmatter).length > 0 && <pre className="frontmatter">{JSON.stringify(frontmatter, null, 2)}</pre>}
        </details>
      )}
      <p className="mono tiny muted">id {r.id}</p>
    </article>
  );
}

/** The text CORTEX extracted from a PDF, page by page, with an in-document search. */
export function PdfPages({ id, doc, initialPage, initialQuery }: { id: string; doc: DocumentInfo; initialPage?: number; initialQuery?: string }) {
  const [page, setPage] = useState(Math.min(Math.max(initialPage ?? 1, 1), Math.max(doc.page_count, 1)));
  const [draft, setDraft] = useState(initialQuery ?? "");
  const [query, setQuery] = useState(initialQuery ?? "");
  const searchable = doc.text_pages > 0;
  const current = useAsync(() => (searchable ? api.page(id, page, query || undefined) : Promise.resolve(null)), [id, page, query, searchable]);
  const matches = current.data?.matches ?? [];
  const prevMatch = [...matches].reverse().find((m) => m < page);
  const nextMatch = matches.find((m) => m > page);
  const go = (n: number) => setPage(Math.min(Math.max(n, 1), doc.page_count));
  const meta = Object.entries(doc.pdf_meta).filter(([k]) => k === "title" || k === "author");

  return (
    <section className="pdf">
      <h4 className="section-title">
        PDF text <span className="muted tiny mono">{doc.extractor}</span>
      </h4>
      {doc.status === "no_text" && <p className="notice">No text layer: the pages are images (a scan). CORTEX does not run OCR, so this PDF cannot be searched inside.</p>}
      {doc.status === "failed" && <p className="notice warn">The text could not be extracted: {doc.error ?? "unknown error"}.</p>}
      {doc.status === "partial" && doc.error && <p className="notice">{doc.error}.</p>}
      <dl className="meta-grid mono">
        <dt>pages</dt>
        <dd>
          {doc.page_count} · text on {doc.text_pages}
        </dd>
        {meta.map(([k, v]) => (
          <Fragment key={k}>
            <dt>{k}</dt>
            <dd className="break">{v}</dd>
          </Fragment>
        ))}
        <dt>file</dt>
        <dd className="break">
          <a href={api.fileUrl(id, page)} target="_blank" rel="noopener noreferrer">
            Open PDF{searchable ? ` at p. ${page}` : ""} ↗
          </a>{" "}
          <span className="muted">{doc.path}</span>
        </dd>
        <dt>read</dt>
        <dd>{fmtDate(doc.extracted_at, true)}</dd>
      </dl>
      {searchable && (
        <>
          <form
            className="page-find"
            onSubmit={(e) => {
              e.preventDefault();
              setQuery(draft.trim());
            }}
          >
            <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="find in this PDF ⏎" aria-label="Find in this PDF" />
          </form>
          {query && (
            <p className="mono tiny muted page-matches">
              {matches.length ? `“${query}” on ${matches.length} page${matches.length === 1 ? "" : "s"}` : `“${query}” is not in the extracted text`}
              {matches.length > 0 && (
                <>
                  {" · "}
                  <button className="link" disabled={prevMatch === undefined} onClick={() => prevMatch !== undefined && go(prevMatch)}>
                    ‹ previous
                  </button>{" "}
                  <button className="link" disabled={nextMatch === undefined} onClick={() => nextMatch !== undefined && go(nextMatch)}>
                    next ›
                  </button>
                </>
              )}
            </p>
          )}
          <div className="page-nav mono tiny">
            <button onClick={() => go(page - 1)} disabled={page <= 1} aria-label="Previous page">
              ‹
            </button>
            <label>
              page <input type="number" min={1} max={doc.page_count} value={page} onChange={(e) => e.target.value && go(Number(e.target.value))} /> of {doc.page_count}
            </label>
            <button onClick={() => go(page + 1)} disabled={page >= doc.page_count} aria-label="Next page">
              ›
            </button>
          </div>
          <div className="page-text" data-testid="page-text">
            {current.error ? <span className="warn">{current.error}</span> : !current.data ? <span className="muted">loading…</span> : current.data.has_text ? <Snippet text={current.data.text} /> : <span className="muted">No text on this page (a figure, a scan or a blank page).</span>}
          </div>
        </>
      )}
    </section>
  );
}

function RelList({ title, items, types, nav, busy, act, arrow }: { title: string; items: RelationshipView[]; types: string[]; nav: Nav; busy: boolean; act: (fn: () => Promise<unknown>, done?: string) => Promise<void>; arrow: string }) {
  if (!items.length) return null;
  return (
    <div className="rel-list">
      <div className="mono tiny muted">{title}</div>
      <ul className="plain">
        {items.map((rel) => (
          <li key={rel.id} className="rel">
            <select value={rel.rel_type} disabled={busy} onChange={(e) => act(() => api.updateRelationship(rel.id, { rel_type: e.target.value }), "Relationship updated.")} aria-label="Relationship type">
              {types.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <span className="mono muted">{arrow}</span>
            <button className="link" onClick={() => nav.openRecord(rel.other.id)}>
              {rel.other.title}
            </button>
            <CategoryChip category={rel.other.category} />
            {rel.origin !== "user" && <span className="mono tiny muted">{rel.origin}</span>}
            <button className="ghost danger-text" title="Remove relationship" disabled={busy} onClick={() => window.confirm(`Remove the "${rel.rel_type}" link to "${rel.other.title}"?`) && act(() => api.deleteRelationship(rel.id), "Link removed.")}>
              ×
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Versions({ id, rev, act, busy }: { id: string; rev: number; act: (fn: () => Promise<unknown>, done?: string) => Promise<void>; busy: boolean }) {
  const versions = useAsync(() => api.versions(id), [id, rev]);
  const [open, setOpen] = useState<number | null>(null);
  if (!versions.data) return <p className="muted tiny mono">loading…</p>;
  if (!versions.data.length) return <p className="muted tiny">No earlier versions.</p>;
  return (
    <ul className="plain versions">
      {versions.data.map((v) => (
        <li key={v.id}>
          <div className="version-head mono tiny">
            <span>{fmtDate(v.created_at, true)}</span>
            <span className="muted">{v.reason}</span>
            <button className="link" onClick={() => setOpen(open === v.id ? null : v.id)}>
              {open === v.id ? "hide" : "show"}
            </button>
            <button className="link" disabled={busy} onClick={() => window.confirm("Restore this version? The current state is saved as a new version first.") && act(() => api.revert(id, v.id), "Version restored.")}>
              restore
            </button>
          </div>
          {open === v.id && (
            <pre className="snapshot">
              {String(v.snapshot.title)}
              {"\n\n"}
              {String(v.snapshot.content ?? "")}
            </pre>
          )}
        </li>
      ))}
    </ul>
  );
}
