import { describe, expect, it } from "vitest";

import { buildRecord, EMPTY_FORM } from "./CapturePanel";

const linked = { id: "p1", title: "Protocol" } as never;

describe("buildRecord", () => {
  it("records a decision with rationale, alternatives and links", () => {
    const r = buildRecord("decision", {
      ...EMPTY_FORM,
      title: "Keep both slab branches",
      body: "Keep slab and no-slab branches until CP1.",
      rationale: "Evidence is not selected yet.",
      alternatives: "- drop the no-slab branch\nslab only",
      projectId: "proj",
      links: [linked],
    });
    expect(r).toMatchObject({ category: "projects", kind: "decision", status: "proposed", project_id: "proj" });
    expect(r.content).toContain("**Decision:** Keep slab");
    expect(r.content).toContain("**Rationale:** Evidence");
    expect(r.content).toContain("- drop the no-slab branch\n- slab only");
    expect(r.metadata).toEqual({ rationale: "Evidence is not selected yet.", alternatives: ["drop the no-slab branch", "slab only"] });
    expect(r.links).toEqual([{ target_id: "p1", rel_type: "relates_to" }]);
  });

  it("captures a session with date, duration and activities", () => {
    const r = buildRecord("session", { ...EMPTY_FORM, title: "IMK review", body: "Went through the four modes.", date: "2026-10-22", minutes: "60", activities: "reading\nquiz" });
    expect(r).toMatchObject({ category: "sessions", kind: "session", occurred_at: "2026-10-22" });
    expect(r.content).toBe("Went through the four modes.\n\n## Activities\n- reading\n- quiz");
    expect(r.metadata).toEqual({ activities: ["reading", "quiz"], duration_minutes: 60 });
  });

  it("captures a structured fact without claiming verification", () => {
    const r = buildRecord("fact", { ...EMPTY_FORM, title: "Column cap", subject: "Perimeter column", predicate: "max width", value: "32 in.", source: "README" });
    expect(r).toMatchObject({ category: "facts", subject: "Perimeter column", predicate: "max width", value: "32 in.", source_ref: "README", verified: false });
  });

  it("puts quick notes in the inbox and never gives a project a parent project", () => {
    expect(buildRecord("note", { ...EMPTY_FORM, title: "Idea", noteKind: "idea" })).toMatchObject({ category: "inbox", kind: "idea" });
    expect(buildRecord("project", { ...EMPTY_FORM, title: "P", projectId: "other", objectives: "a\nb" })).toMatchObject({
      project_id: null,
      status: "active",
      metadata: { objectives: ["a", "b"] },
    });
  });
});
