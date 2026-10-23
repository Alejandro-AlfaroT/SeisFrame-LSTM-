import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Nav } from "../App";
import type { ConflictDetail, MemoryRecord } from "../types";

const record = (id: string, title: string, value: string): MemoryRecord => ({
  id,
  title,
  content: "",
  category: "facts",
  region: "hippocampus",
  kind: "fact",
  status: null,
  project_id: null,
  project: null,
  source_ref: `source for ${title}`,
  source_hash: null,
  confidence: 0.7,
  verified: false,
  verified_at: null,
  subject: "Tower",
  predicate: "height",
  value,
  occurred_at: null,
  metadata: {},
  is_demo: false,
  deleted: false,
  deleted_at: null,
  created_at: "2026-10-22T10:00:00Z",
  updated_at: "2026-10-22T10:00:00Z",
  tags: [],
  relationships: { outgoing: [], incoming: [] },
  version_count: 0,
  conflicts: [],
});

const detail: ConflictDetail = {
  id: "c1",
  status: "open",
  reason: "structured",
  explanation: "Both facts describe 'Tower · height' with different values.",
  score: 0.95,
  excerpt_a: "Tower · height = 31 m",
  excerpt_b: "Tower · height = 34 m",
  resolution: null,
  resolution_note: null,
  detected_at: "2026-10-22T10:00:00Z",
  resolved_at: null,
  record_a: record("a", "Brochure", "31 m"),
  record_b: record("b", "Survey", "34 m"),
};

const mocks = vi.hoisted(() => ({
  conflict: vi.fn(),
  conflicts: vi.fn(),
  resolveConflict: vi.fn(),
  reopenConflict: vi.fn(),
  scanConflicts: vi.fn(),
}));
vi.mock("../api", () => ({ api: mocks }));

import ConflictPanel from "./ConflictPanel";

const nav: Nav = { go: vi.fn(), openRecord: vi.fn(), openRef: vi.fn(), changed: vi.fn(), toast: vi.fn() };

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("conflict review", () => {
  it("shows both sources and records only the decision the user picks", async () => {
    mocks.conflict.mockResolvedValue(detail);
    mocks.resolveConflict.mockResolvedValue({ ...detail, status: "resolved" });
    render(<ConflictPanel conflictId="c1" rev={0} nav={nav} onFocus={() => {}} />);

    expect(await screen.findByText("Tower · height = 31 m")).toBeTruthy();
    expect(screen.getByText("Tower · height = 34 m")).toBeTruthy();
    expect(screen.getByText("source for Brochure")).toBeTruthy();

    const submit = screen.getByText("Record decision") as HTMLButtonElement;
    expect(submit.disabled).toBe(true); // nothing is resolved without an explicit choice

    fireEvent.click(screen.getByText("Prefer B"));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "survey is newer" } });
    fireEvent.click(submit);
    await waitFor(() => expect(mocks.resolveConflict).toHaveBeenCalledWith("c1", "prefer_b", "survey is newer"));
    expect(nav.changed).toHaveBeenCalled();
  });

  it("offers to reopen a resolved conflict", async () => {
    mocks.conflict.mockResolvedValue({ ...detail, status: "resolved", resolution: "keep_both", resolved_at: "2026-10-22T11:00:00Z" });
    mocks.reopenConflict.mockResolvedValue(detail);
    render(<ConflictPanel conflictId="c1" rev={0} nav={nav} onFocus={() => {}} />);
    fireEvent.click(await screen.findByText("Reopen"));
    await waitFor(() => expect(mocks.reopenConflict).toHaveBeenCalledWith("c1"));
  });
});
