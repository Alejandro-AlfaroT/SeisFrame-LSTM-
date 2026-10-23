import { useState } from "react";

import { api } from "../api";
import type { Nav } from "../App";
import { confirmTyped, errorText, fmtDate, useAsync } from "../lib";
import { CategoryChip } from "./common";

export default function DataPanel({ rev, nav, includeDemo, setIncludeDemo }: { rev: number; nav: Nav; includeDemo: boolean; setIncludeDemo: (v: boolean) => void }) {
  const health = useAsync(() => api.health(), []);
  const backups = useAsync(() => api.backups(), [rev]);
  const trash = useAsync(() => api.trash(), [rev]);
  const stats = useAsync(() => api.stats(), [rev]);
  const [busy, setBusy] = useState(false);

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
    <div className="data">
      <h3 className="panel-title">Data, backups and trash</h3>
      <dl className="meta-grid mono tiny">
        <dt>database</dt>
        <dd className="break">{health.data?.database ?? "…"}</dd>
        <dt>schema</dt>
        <dd>v{health.data?.schema_version ?? "?"}</dd>
        <dt>network</dt>
        <dd>localhost only · no external APIs</dd>
      </dl>

      <h4 className="section-title">Backup</h4>
      <p className="tiny muted">A consistent copy of the SQLite file, safe to take while the app runs. Schema migrations also back up automatically first.</p>
      <div className="toolbar">
        <button className="primary" disabled={busy} onClick={() => run(() => api.backup("manual"), "Backup written.")}>
          Back up now
        </button>
      </div>
      <ul className="plain mono tiny">
        {(backups.data ?? []).slice(0, 8).map((b) => (
          <li key={b.name} title={b.path}>
            {b.name} · {(b.bytes / 1024).toFixed(0)} KB
          </li>
        ))}
        {backups.data?.length === 0 && <li className="muted">No backups yet.</li>}
      </ul>
      <p className="tiny muted">
        To restore: stop the server, then run <code>python -m cortex_server restore &lt;file&gt; --yes</code> (the current database is saved first).
      </p>

      <h4 className="section-title">Export</h4>
      <div className="toolbar">
        <a className="button" href="/api/export/json" download>
          Export JSON (everything)
        </a>
        <a className="button" href="/api/export/markdown" download>
          Export Markdown (.zip)
        </a>
      </div>

      <h4 className="section-title">Trash ({trash.data?.length ?? 0})</h4>
      <p className="tiny muted">Trashed records stay in the database until you delete them permanently here.</p>
      <ul className="results">
        {(trash.data ?? []).map((r) => (
          <li key={r.id} className="record-row">
            <button className="row-main" onClick={() => nav.openRecord(r.id)}>
              <span className="row-title">{r.title}</span>
              <span className="row-meta mono">
                <CategoryChip category={r.category} /> trashed {fmtDate(r.deleted_at, true)}
              </span>
            </button>
            <div className="row-actions">
              <button disabled={busy} onClick={() => run(() => api.restoreRecord(r.id), "Restored.")}>
                Restore
              </button>
              <button
                className="danger"
                disabled={busy}
                onClick={() =>
                  confirmTyped(`Permanently delete “${r.title}”, its links and its version history? This cannot be undone. Make a backup first if unsure.`, "delete") &&
                  run(() => api.purgeRecord(r.id), "Deleted permanently.")
                }
              >
                Delete forever
              </button>
            </div>
          </li>
        ))}
      </ul>

      <h4 className="section-title">Demonstration data</h4>
      <p className="tiny muted">
        Eleven fictional records about a museum exhibit, flagged DEMO and unrelated to SeisFrame, including two deliberate contradictions to try the conflict review with.
      </p>
      <div className="toolbar">
        <button disabled={busy || (stats.data?.demo_records ?? 0) > 0} onClick={() => run(() => api.loadDemo(), "Demo data loaded.")}>
          Load demo data
        </button>
        <button
          className="danger"
          disabled={busy || (stats.data?.demo_records ?? 0) === 0}
          onClick={() => window.confirm("Remove all DEMO records? Only records flagged as demo are affected.") && run(() => api.purgeDemo(), "Demo data removed.")}
        >
          Remove demo data
        </button>
        <label className="check">
          <input type="checkbox" checked={includeDemo} onChange={(e) => setIncludeDemo(e.target.checked)} /> show demo in brain
        </label>
      </div>
    </div>
  );
}
