import { useState } from "react";

import { api, type NewRecord } from "../api";
import type { Nav } from "../App";
import { errorText } from "../lib";
import type { RecordSummary } from "../types";
import { CategoryChip, ProjectSelect, RecordPicker, TagInput } from "./common";

export type CaptureMode = "note" | "fact" | "decision" | "session" | "project" | "milestone";

const MODES: { mode: CaptureMode; label: string; hint: string }[] = [
  { mode: "note", label: "Note", hint: "Quick capture into the Inbox; file it later." },
  { mode: "fact", label: "Fact", hint: "Stable knowledge with a source and confidence." },
  { mode: "decision", label: "Decision", hint: "A project decision with rationale and alternatives." },
  { mode: "session", label: "Session", hint: "Summary of a work session." },
  { mode: "project", label: "Project", hint: "An active project with objectives." },
  { mode: "milestone", label: "Milestone", hint: "A dated target inside a project." },
];

const lines = (text: string) =>
  text
    .split("\n")
    .map((l) => l.replace(/^\s*[-*]\s*/, "").trim())
    .filter(Boolean);

export interface CaptureForm {
  title: string;
  body: string;
  tags: string[];
  projectId: string | null;
  links: RecordSummary[];
  noteKind: string;
  subject: string;
  predicate: string;
  value: string;
  source: string;
  confidence: number;
  verified: boolean;
  status: string;
  date: string;
  rationale: string;
  alternatives: string;
  minutes: string;
  activities: string;
  objectives: string;
}

export const EMPTY_FORM: CaptureForm = {
  title: "",
  body: "",
  tags: [],
  projectId: null,
  links: [],
  noteKind: "note",
  subject: "",
  predicate: "",
  value: "",
  source: "",
  confidence: 0.7,
  verified: false,
  status: "",
  date: "",
  rationale: "",
  alternatives: "",
  minutes: "",
  activities: "",
  objectives: "",
};

/** Turn the form for one capture mode into a record payload. Exported for tests. */
export function buildRecord(mode: CaptureMode, f: CaptureForm): NewRecord {
  const base = {
    title: f.title.trim(),
    tags: f.tags,
    project_id: f.projectId,
    links: f.links.map((l) => ({ target_id: l.id, rel_type: "relates_to" })),
  };
  switch (mode) {
    case "note":
      return { ...base, category: "inbox", kind: f.noteKind, content: f.body };
    case "fact":
      return {
        ...base,
        category: "facts",
        kind: "fact",
        content: f.body,
        subject: f.subject || null,
        predicate: f.predicate || null,
        value: f.value || null,
        source_ref: f.source || null,
        confidence: f.confidence,
        verified: f.verified,
      };
    case "decision": {
      const parts = [`**Decision:** ${f.body.trim()}`];
      if (f.rationale.trim()) parts.push(`**Rationale:** ${f.rationale.trim()}`);
      if (lines(f.alternatives).length) parts.push(`**Alternatives considered:**\n${lines(f.alternatives).map((a) => `- ${a}`).join("\n")}`);
      return {
        ...base,
        category: "projects",
        kind: "decision",
        status: f.status || "proposed",
        occurred_at: f.date || null,
        content: parts.join("\n\n"),
        metadata: { rationale: f.rationale.trim(), alternatives: lines(f.alternatives) },
      };
    }
    case "session": {
      const acts = lines(f.activities);
      const content = acts.length ? `${f.body.trim()}\n\n## Activities\n${acts.map((a) => `- ${a}`).join("\n")}` : f.body.trim();
      return {
        ...base,
        category: "sessions",
        kind: "session",
        occurred_at: f.date || new Date().toISOString().slice(0, 10),
        content,
        metadata: { activities: acts, ...(f.minutes ? { duration_minutes: Number(f.minutes) } : {}) },
      };
    }
    case "project":
      return {
        ...base,
        project_id: null,
        category: "projects",
        kind: "project",
        status: f.status || "active",
        content: f.body,
        metadata: { objectives: lines(f.objectives) },
      };
    case "milestone":
      return { ...base, category: "projects", kind: "milestone", status: f.status || "planned", occurred_at: f.date || null, content: f.body };
  }
}

export default function CapturePanel({ initialMode, nav }: { initialMode: CaptureMode; nav: Nav }) {
  const [mode, setMode] = useState<CaptureMode>(initialMode);
  const [f, setF] = useState<CaptureForm>(EMPTY_FORM);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const set = <K extends keyof CaptureForm>(k: K, v: CaptureForm[K]) => setF((prev) => ({ ...prev, [k]: v }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "milestone" && !f.projectId) {
      setErr("A milestone needs a project.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const created = await api.createRecord(buildRecord(mode, f));
      nav.changed();
      nav.toast(`Saved “${created.title}”.`);
      setF(EMPTY_FORM);
      nav.openRecord(created.id);
    } catch (ex) {
      setErr(errorText(ex));
    } finally {
      setBusy(false);
    }
  };

  const bodyLabel: Record<CaptureMode, string> = {
    note: "Note",
    fact: "Statement / details",
    decision: "What was decided",
    session: "Summary",
    project: "Description",
    milestone: "Definition of done",
  };

  return (
    <form className="capture" onSubmit={submit}>
      <div className="seg wrap">
        {MODES.map((m) => (
          <button type="button" key={m.mode} className={mode === m.mode ? "active" : ""} onClick={() => setMode(m.mode)}>
            {m.label}
          </button>
        ))}
      </div>
      <p className="tiny muted">{MODES.find((m) => m.mode === mode)?.hint}</p>

      <label className="field">
        <span>Title</span>
        <input required value={f.title} onChange={(e) => set("title", e.target.value)} />
      </label>

      {mode === "note" && (
        <label className="field">
          <span>Type</span>
          <select value={f.noteKind} onChange={(e) => set("noteKind", e.target.value)}>
            <option value="note">note</option>
            <option value="idea">idea</option>
            <option value="observation">observation</option>
            <option value="question">question</option>
          </select>
        </label>
      )}

      {mode === "fact" && (
        <>
          <div className="field-row">
            <label className="field">
              <span>Subject</span>
              <input value={f.subject} onChange={(e) => set("subject", e.target.value)} placeholder="Perimeter column" />
            </label>
            <label className="field">
              <span>Predicate</span>
              <input value={f.predicate} onChange={(e) => set("predicate", e.target.value)} placeholder="max width" />
            </label>
            <label className="field">
              <span>Value</span>
              <input value={f.value} onChange={(e) => set("value", e.target.value)} placeholder="32 in." />
            </label>
          </div>
          <p className="tiny muted">Optional. Facts with the same subject and predicate but different values are flagged for review.</p>
          <label className="field">
            <span>Source</span>
            <input value={f.source} onChange={(e) => set("source", e.target.value)} placeholder="DOI, URL, file, page, section" />
          </label>
          <div className="field-row">
            <label className="field">
              <span>Confidence {f.confidence.toFixed(2)}</span>
              <input type="range" min={0} max={1} step={0.05} value={f.confidence} onChange={(e) => set("confidence", Number(e.target.value))} />
            </label>
            <label className="check">
              <input type="checkbox" checked={f.verified} onChange={(e) => set("verified", e.target.checked)} /> I have verified this against the source
            </label>
          </div>
        </>
      )}

      {(mode === "decision" || mode === "session" || mode === "milestone") && (
        <div className="field-row">
          <label className="field">
            <span>{mode === "milestone" ? "Due" : "Date"}</span>
            <input type="date" value={f.date} onChange={(e) => set("date", e.target.value)} />
          </label>
          {mode !== "session" && (
            <label className="field">
              <span>Status</span>
              <select value={f.status} onChange={(e) => set("status", e.target.value)}>
                {(mode === "decision" ? ["proposed", "accepted", "superseded"] : ["planned", "in-progress", "complete", "blocked"]).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
          )}
          {mode === "session" && (
            <label className="field">
              <span>Minutes</span>
              <input type="number" min={0} value={f.minutes} onChange={(e) => set("minutes", e.target.value)} />
            </label>
          )}
        </div>
      )}

      {mode === "project" && (
        <label className="field">
          <span>Status</span>
          <select value={f.status || "active"} onChange={(e) => set("status", e.target.value)}>
            {["active", "paused", "complete"].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      )}

      <label className="field">
        <span>{bodyLabel[mode]}</span>
        <textarea rows={mode === "note" ? 8 : 5} value={f.body} onChange={(e) => set("body", e.target.value)} />
      </label>

      {mode === "decision" && (
        <>
          <label className="field">
            <span>Rationale</span>
            <textarea rows={3} value={f.rationale} onChange={(e) => set("rationale", e.target.value)} />
          </label>
          <label className="field">
            <span>Alternatives considered (one per line)</span>
            <textarea rows={3} value={f.alternatives} onChange={(e) => set("alternatives", e.target.value)} />
          </label>
        </>
      )}
      {mode === "session" && (
        <label className="field">
          <span>Activities (one per line)</span>
          <textarea rows={4} value={f.activities} onChange={(e) => set("activities", e.target.value)} />
        </label>
      )}
      {mode === "project" && (
        <label className="field">
          <span>Objectives (one per line)</span>
          <textarea rows={3} value={f.objectives} onChange={(e) => set("objectives", e.target.value)} />
        </label>
      )}

      {mode !== "project" && (
        <label className="field">
          <span>Project{mode === "milestone" ? " (required)" : ""}</span>
          <ProjectSelect value={f.projectId} onChange={(v) => set("projectId", v)} />
        </label>
      )}

      <div className="field">
        <span>Tags</span>
        <TagInput value={f.tags} onChange={(t) => set("tags", t)} />
      </div>

      <div className="field">
        <span>{mode === "session" ? "Related documents" : "Link related records"}</span>
        <RecordPicker exclude={f.links.map((l) => l.id)} onPick={(r) => set("links", [...f.links, r])} />
        <ul className="plain">
          {f.links.map((l) => (
            <li key={l.id} className="rel">
              <span className="mono muted">relates_to →</span> {l.title} <CategoryChip category={l.category} />
              <button type="button" className="ghost" onClick={() => set("links", f.links.filter((x) => x.id !== l.id))}>
                ×
              </button>
            </li>
          ))}
        </ul>
      </div>

      {err && <p className="notice warn">{err}</p>}
      <button className="primary" type="submit" disabled={busy || !f.title.trim()}>
        Save {MODES.find((m) => m.mode === mode)?.label.toLowerCase()}
      </button>
    </form>
  );
}
