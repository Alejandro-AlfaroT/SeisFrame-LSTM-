import { useRef, useState } from "react";

import { api } from "../api";
import type { Nav } from "../App";
import { errorText, fmtDate, useAsync } from "../lib";
import type { ImportReport } from "../types";
import { ProjectSelect } from "./common";

const MAX_FILE_BYTES = 2 * 1024 * 1024;

interface Picked {
  path: string;
  content: string;
}

/** Relative path of an uploaded file; folder uploads keep their sub-folders. */
export function relativePath(file: File & { webkitRelativePath?: string }): { label: string; path: string } {
  const rel = file.webkitRelativePath || file.name;
  const parts = rel.split("/");
  if (parts.length > 1) return { label: parts[0], path: parts.slice(1).join("/") };
  return { label: "upload", path: rel };
}

export default function ImportPanel({ nav }: { nav: Nav }) {
  const [files, setFiles] = useState<Picked[]>([]);
  const [label, setLabel] = useState("upload");
  const [route, setRoute] = useState<"inbox" | "by_type">("inbox");
  const [projectId, setProjectId] = useState<string | null>(null);
  const [includeTemplates, setIncludeTemplates] = useState(false);
  const [updateEdited, setUpdateEdited] = useState(false);
  const [preview, setPreview] = useState<ImportReport | null>(null);
  const [result, setResult] = useState<ImportReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [tooBig, setTooBig] = useState<string[]>([]);
  const batches = useAsync(() => api.imports(), [result]);
  const folderInput = useRef<HTMLInputElement>(null);

  const pick = async (list: FileList | null) => {
    if (!list) return;
    setPreview(null);
    setResult(null);
    const picked: Picked[] = [];
    const big: string[] = [];
    let firstLabel: string | null = null;
    for (const file of Array.from(list)) {
      if (!/\.(md|markdown)$/i.test(file.name)) continue;
      const { label: l, path } = relativePath(file);
      firstLabel ??= l;
      if (file.size > MAX_FILE_BYTES) {
        big.push(path);
        continue;
      }
      picked.push({ path, content: await file.text() });
    }
    setFiles(picked);
    setTooBig(big);
    setLabel(firstLabel ?? "upload");
  };

  const run = async (dryRun: boolean) => {
    setBusy(true);
    setErr(null);
    try {
      const report = await api.importMarkdown({
        files,
        route,
        project_id: projectId,
        source_label: label,
        dry_run: dryRun,
        include_templates: includeTemplates,
        update_edited: updateEdited,
      });
      if (dryRun) setPreview(report);
      else {
        setResult(report);
        setPreview(null);
        nav.changed();
        nav.toast(`Imported ${report.counts.created ?? 0} new, ${report.counts.updated ?? 0} updated.`);
      }
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const report = result ?? preview;
  return (
    <div className="import">
      <h3 className="panel-title">Import Markdown</h3>
      <p className="tiny muted">
        Imports only happen when you press <b>Import</b>. Files are treated as untrusted: stored as text, scripts never run, remote images never load, nothing is marked verified. Hidden folders
        are skipped.
      </p>
      <div className="toolbar">
        <label className="button">
          Choose files
          <input type="file" accept=".md,.markdown,text/markdown" multiple hidden onChange={(e) => pick(e.target.files)} />
        </label>
        <button
          onClick={() => {
            folderInput.current?.setAttribute("webkitdirectory", "");
            folderInput.current?.click();
          }}
        >
          Choose folder
        </button>
        <input ref={folderInput} type="file" multiple hidden onChange={(e) => pick(e.target.files)} />
      </div>
      {files.length > 0 && (
        <p className="mono tiny">
          {files.length} Markdown file{files.length === 1 ? "" : "s"} · {(files.reduce((n, f) => n + f.content.length, 0) / 1024).toFixed(0)} KB
          {tooBig.length > 0 && <span className="warn"> · {tooBig.length} over 2 MB skipped</span>}
        </p>
      )}

      <div className="field">
        <span>Source label (prefix for source references)</span>
        <input value={label} onChange={(e) => setLabel(e.target.value)} />
      </div>
      <div className="field">
        <span>Where should notes go?</span>
        <label className="check">
          <input type="radio" checked={route === "inbox"} onChange={() => setRoute("inbox")} /> Inbox, for triage (a category is suggested for each note)
        </label>
        <label className="check">
          <input type="radio" checked={route === "by_type"} onChange={() => setRoute("by_type")} /> File by note type / folder (decision → Projects, daily → Sessions, literature → Facts…)
        </label>
      </div>
      <div className="field">
        <span>Attach to project</span>
        <ProjectSelect value={projectId} onChange={setProjectId} />
      </div>
      <label className="check">
        <input type="checkbox" checked={includeTemplates} onChange={(e) => setIncludeTemplates(e.target.checked)} /> include the Templates folder
      </label>
      <label className="check">
        <input type="checkbox" checked={updateEdited} onChange={(e) => setUpdateEdited(e.target.checked)} /> also refresh records I edited in CORTEX (old versions are kept)
      </label>

      <div className="toolbar">
        <button disabled={busy || !files.length} onClick={() => run(true)}>
          Preview (dry run)
        </button>
        <button className="primary" disabled={busy || !preview} onClick={() => run(false)} title={preview ? "" : "Preview first"}>
          Import {preview ? `${(preview.counts.created ?? 0) + (preview.counts.updated ?? 0)} change(s)` : ""}
        </button>
      </div>
      {err && <p className="notice warn">{err}</p>}

      {report && (
        <div className="report">
          <h4 className="section-title">{report.dry_run ? "Preview: nothing written yet" : "Imported"}</h4>
          <p className="mono tiny">
            {Object.entries(report.counts)
              .map(([k, v]) => `${v} ${k}`)
              .join(" · ") || "no files"}{" "}
            · links {report.links.resolved} resolved / {report.links.unresolved} unresolved
            {report.conflicts && ` · ${report.conflicts.new} new conflict candidate(s)`}
          </p>
          {report.skipped.length > 0 && (
            <details>
              <summary className="mono tiny">{report.skipped.length} skipped</summary>
              <ul className="plain mono tiny">
                {report.skipped.map((s) => (
                  <li key={s.path}>
                    {s.path} — {s.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}
          <table className="report-table mono tiny">
            <thead>
              <tr>
                <th>file</th>
                <th>action</th>
                <th>→</th>
              </tr>
            </thead>
            <tbody>
              {report.files.map((f) => (
                <tr key={f.path}>
                  <td title={f.title}>
                    {!report.dry_run && f.record_id ? (
                      <button className="link" onClick={() => nav.openRecord(f.record_id!)}>
                        {f.path}
                      </button>
                    ) : (
                      f.path
                    )}
                    {f.warnings.length > 0 && <div className="warn">{f.warnings.join("; ")}</div>}
                  </td>
                  <td>{f.action}</td>
                  <td>
                    {f.category}/{f.kind}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h4 className="section-title">Past imports</h4>
      <ul className="plain mono tiny">
        {(batches.data ?? []).map((b) => {
          let summary = "";
          try {
            const s = JSON.parse(b.stats) as { counts?: Record<string, number> };
            summary = Object.entries(s.counts ?? {})
              .map(([k, v]) => `${v} ${k}`)
              .join(", ");
          } catch {
            summary = "";
          }
          return (
            <li key={b.id}>
              {fmtDate(b.created_at, true)} · {b.route} · {summary} <span className="muted break">{b.source}</span>
            </li>
          );
        })}
      </ul>
      <p className="tiny muted">
        Large folders can also be imported from the command line: <code>python -m cortex_server import-md "&lt;folder&gt;" --dry-run</code>
      </p>
    </div>
  );
}
