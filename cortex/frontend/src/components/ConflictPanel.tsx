import { useEffect, useState } from "react";

import { api } from "../api";
import type { Nav } from "../App";
import { errorText, fmtDate, useAsync } from "../lib";
import type { ConflictDetail, MemoryRecord } from "../types";
import { Badges, CategoryChip } from "./common";

const RESOLUTIONS: { value: string; label: string; help: string }[] = [
  { value: "prefer_a", label: "Prefer A", help: "A supersedes B: adds a 'supersedes' link and marks B as superseded. B's content is kept." },
  { value: "prefer_b", label: "Prefer B", help: "B supersedes A: adds a 'supersedes' link and marks A as superseded. A's content is kept." },
  { value: "keep_both", label: "Keep both", help: "Both stand, e.g. they hold under different conditions. Nothing is changed." },
  { value: "not_a_conflict", label: "Not a conflict", help: "False alarm. The detector will not raise this pair of statements again." },
];

export default function ConflictPanel({ conflictId, rev, nav, onFocus }: { conflictId?: string; rev: number; nav: Nav; onFocus: (ids: string[]) => void }) {
  const [status, setStatus] = useState<string>("open");
  const list = useAsync(() => (conflictId ? Promise.resolve(null) : api.conflicts(status === "all" ? undefined : status)), [status, rev, conflictId]);
  const [scanning, setScanning] = useState(false);

  useEffect(() => {
    if (!conflictId && list.data) onFocus(list.data.flatMap((c) => [c.a.id, c.b.id]));
  }, [conflictId, list.data, onFocus]);

  if (conflictId) return <ConflictReview id={conflictId} rev={rev} nav={nav} onFocus={onFocus} />;

  const scan = async () => {
    setScanning(true);
    try {
      const r = await api.scanConflicts();
      nav.changed();
      nav.toast(`Scanned ${r.scanned_records} records (${r.statements} statements): ${r.new_conflicts.length} new candidate(s).`);
    } catch (e) {
      nav.toast(errorText(e));
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="conflicts">
      <h3 className="panel-title">Conflict review</h3>
      <p className="tiny muted">
        Candidates come from local heuristics: facts with the same subject and predicate but different values, and near-identical statements that differ in a number, a negation, or an opposing
        term. They are suggestions. Nothing is deleted or overwritten, and only you resolve them.
      </p>
      <div className="toolbar">
        <div className="seg">
          {["open", "resolved", "dismissed", "all"].map((s) => (
            <button key={s} className={status === s ? "active" : ""} onClick={() => setStatus(s)}>
              {s}
            </button>
          ))}
        </div>
        <button onClick={scan} disabled={scanning}>
          {scanning ? "Scanning…" : "Scan now"}
        </button>
      </div>
      {list.error && <p className="notice warn">{list.error}</p>}
      {list.data?.length === 0 && <p className="muted">No {status === "all" ? "" : status} conflicts.</p>}
      <ul className="results">
        {list.data?.map((c) => (
          <li key={c.id}>
            <button className="row-main" onClick={() => nav.go({ kind: "conflicts", id: c.id })}>
              <span className="row-meta mono">
                <span className={`chip ${c.status === "open" ? "warn" : ""}`}>{c.status}</span> {c.reason} · score {c.score.toFixed(2)}
                {c.is_demo && <span className="chip demo">DEMO</span>}
              </span>
              <span className="row-title">
                {c.a.title} <span className="muted">vs</span> {c.b.title}
              </span>
              <span className="row-excerpt">{c.explanation}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Side({ label, record, excerpt, nav }: { label: string; record: MemoryRecord; excerpt: string; nav: Nav }) {
  return (
    <div className="side">
      <div className="side-label mono">{label}</div>
      <button className="link side-title" onClick={() => nav.openRecord(record.id)}>
        {record.title}
      </button>
      <div className="record-chips">
        <CategoryChip category={record.category} />
        <span className="chip">{record.kind}</span>
        <Badges r={{ ...record, superseded_by: (record.metadata.superseded_by as string | undefined) ?? null }} />
      </div>
      <blockquote className="excerpt">{excerpt}</blockquote>
      <dl className="meta-grid mono tiny">
        <dt>source</dt>
        <dd className="break">{record.source_ref ?? "—"}</dd>
        <dt>confidence</dt>
        <dd>{record.confidence?.toFixed(2) ?? "—"}</dd>
        <dt>date</dt>
        <dd>{fmtDate(record.occurred_at)}</dd>
        <dt>created</dt>
        <dd>{fmtDate(record.created_at, true)}</dd>
        <dt>updated</dt>
        <dd>{fmtDate(record.updated_at, true)}</dd>
      </dl>
    </div>
  );
}

function ConflictReview({ id, rev, nav, onFocus }: { id: string; rev: number; nav: Nav; onFocus: (ids: string[]) => void }) {
  const detail = useAsync(() => api.conflict(id), [id, rev]);
  const [choice, setChoice] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (detail.data) onFocus([detail.data.record_a.id, detail.data.record_b.id]);
  }, [detail.data, onFocus]);

  if (detail.error) return <p className="notice warn">{detail.error}</p>;
  const c: ConflictDetail | undefined = detail.data;
  if (!c) return <p className="muted mono">loading…</p>;

  const run = async (fn: () => Promise<unknown>, msg: string) => {
    setBusy(true);
    setErr(null);
    try {
      await fn();
      nav.changed();
      nav.toast(msg);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="conflict-review">
      <button className="link tiny" onClick={() => nav.go({ kind: "conflicts" })}>
        ← all conflicts
      </button>
      <h3 className="panel-title">
        {c.reason} conflict <span className={`chip ${c.status === "open" ? "warn" : ""}`}>{c.status}</span>
      </h3>
      <p className="explanation">{c.explanation}</p>
      <p className="mono tiny muted">
        detected {fmtDate(c.detected_at, true)} · heuristic score {c.score.toFixed(2)} · review required
      </p>
      <div className="sides">
        <Side label="A" record={c.record_a} excerpt={c.excerpt_a} nav={nav} />
        <Side label="B" record={c.record_b} excerpt={c.excerpt_b} nav={nav} />
      </div>

      {c.status === "open" ? (
        <div className="resolve">
          <h4 className="section-title">Your decision</h4>
          {RESOLUTIONS.map((r) => (
            <label key={r.value} className={`choice${choice === r.value ? " on" : ""}`}>
              <input type="radio" name="resolution" checked={choice === r.value} onChange={() => setChoice(r.value)} />
              <span>
                <b>{r.label}</b>
                <span className="tiny muted"> {r.help}</span>
              </span>
            </label>
          ))}
          <label className="field">
            <span>Note (why)</span>
            <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)} />
          </label>
          {err && <p className="notice warn">{err}</p>}
          <button className="primary" disabled={!choice || busy} onClick={() => run(() => api.resolveConflict(c.id, choice!, note), "Decision recorded.")}>
            Record decision
          </button>
        </div>
      ) : (
        <div className="resolve">
          <p>
            Resolved as <b>{c.resolution?.replace(/_/g, " ")}</b> on {fmtDate(c.resolved_at, true)}.
          </p>
          {c.resolution_note && <blockquote className="excerpt">{c.resolution_note}</blockquote>}
          {err && <p className="notice warn">{err}</p>}
          <button disabled={busy} onClick={() => run(() => api.reopenConflict(c.id), "Reopened; the link and marker the resolution added were removed.")}>
            Reopen
          </button>
        </div>
      )}
    </div>
  );
}
