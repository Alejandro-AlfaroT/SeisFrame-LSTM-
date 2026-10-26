import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Nav } from "../App";
import type { DocumentHit, RecordSummary, SearchResult } from "../types";

const summary = (id: string, title: string, kind: string): RecordSummary => ({
  id,
  title,
  category: "facts",
  region: "hippocampus",
  kind,
  status: null,
  project_id: null,
  occurred_at: null,
  created_at: "2026-10-24T10:00:00Z",
  updated_at: "2026-10-24T10:00:00Z",
  verified: false,
  is_demo: false,
  deleted: false,
  tags: [],
  excerpt: "",
  suggested_category: null,
  superseded_by: null,
});

const ibarra: DocumentHit = {
  ...summary("pdf1", "Ibarra 2005 - Hysteretic Deterioration Models", "pdf"),
  page_count: 23,
  pages_matched: 9,
  passages: [
    { page: 12, snippet: "loading \u0002stiffness\u0003 deterioration", score: 4.1 },
    { page: 3, snippet: "unloading \u0002stiffness\u0003", score: 3.2 },
  ],
};

const result = (notes: RecordSummary[]): SearchResult => ({
  mode: "fulltext",
  total: notes.length,
  results: notes,
  documents: { total_pages: 9, total_documents: 1, capped: false, results: [ibarra] },
});

const mocks = vi.hoisted(() => ({ search: vi.fn(), tags: vi.fn(), projects: vi.fn() }));
vi.mock("../api", () => ({ api: mocks }));

import SearchPanel from "./SearchPanel";

const nav: Nav = { go: vi.fn(), openRecord: vi.fn(), openRef: vi.fn(), changed: vi.fn(), toast: vi.fn() };

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("search inside PDFs", () => {
  it("shows matching pages when no note matches, and opens the PDF at the clicked page", async () => {
    mocks.search.mockResolvedValue(result([]));
    mocks.tags.mockResolvedValue([]);
    mocks.projects.mockResolvedValue([]);
    const onResults = vi.fn();
    render(<SearchPanel initialQuery="stiffness" nav={nav} onResults={onResults} />);

    const page = await screen.findByText("p. 3");
    expect(screen.getByText("9 of 23 pages match", { exact: false })).toBeTruthy();
    expect(screen.getAllByText("stiffness")[0].tagName).toBe("MARK");
    await waitFor(() => expect(onResults).toHaveBeenCalledWith(["pdf1"]));

    fireEvent.click(page);
    expect(nav.openRecord).toHaveBeenCalledWith("pdf1", { page: 3, q: "stiffness" });
  });

  it("lists notes first and switches to PDF passages on request", async () => {
    mocks.search.mockResolvedValue(result([summary("n1", "IMK calibration note", "note")]));
    mocks.tags.mockResolvedValue([]);
    mocks.projects.mockResolvedValue([]);
    render(<SearchPanel initialQuery="stiffness" nav={nav} onResults={() => {}} />);

    expect(await screen.findByText("IMK calibration note")).toBeTruthy();
    expect(screen.queryByText("p. 12")).toBeNull();
    fireEvent.click(screen.getByText("In PDFs"));
    expect(screen.getByText("p. 12")).toBeTruthy();
    expect(screen.queryByText("IMK calibration note")).toBeNull();
  });
});
