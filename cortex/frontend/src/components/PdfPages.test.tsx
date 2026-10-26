import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DocumentInfo, DocumentPage } from "../types";

const doc = (over: Partial<DocumentInfo> = {}): DocumentInfo => ({
  path: "C:/vault/08 Assets/Literature/Ibarra 2005.pdf",
  sha256: "ab",
  bytes: 1000,
  page_count: 23,
  text_pages: 23,
  failed_pages: 0,
  status: "ok",
  error: null,
  pdf_meta: { title: "Hysteretic models", author: "Ibarra, Medina, Krawinkler" },
  extractor: "pypdf-6/cortex-1",
  extracted_at: "2026-10-24T10:00:00Z",
  ...over,
});

const pageOf = (page: number, q: string | undefined): DocumentPage => ({
  record_id: "pdf1",
  page,
  page_count: 23,
  has_text: true,
  text: q ? `page ${page}: loading \u0002${q}\u0003` : `page ${page}: plain text`,
  matches: q ? [3, 12, 20] : [],
});

const mocks = vi.hoisted(() => ({ page: vi.fn(), fileUrl: vi.fn((id: string, page?: number) => `/api/records/${id}/file#page=${page}`) }));
vi.mock("../api", () => ({ api: mocks }));

import { PdfPages } from "./RecordPanel";

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("PDF text in a record", () => {
  it("opens at the requested page with the query marked, and steps between matching pages", async () => {
    mocks.page.mockImplementation(async (_id: string, page: number, q?: string) => pageOf(page, q));
    render(<PdfPages id="pdf1" doc={doc()} initialPage={12} initialQuery="stiffness" />);

    expect(await screen.findByText("page 12: loading", { exact: false })).toBeTruthy();
    expect(mocks.page).toHaveBeenLastCalledWith("pdf1", 12, "stiffness");
    expect(screen.getByText("stiffness").tagName).toBe("MARK");
    expect(screen.getByText("“stiffness” on 3 pages", { exact: false })).toBeTruthy();
    expect(screen.getByText("Open PDF at p. 12 ↗").getAttribute("href")).toBe("/api/records/pdf1/file#page=12");

    fireEvent.click(screen.getByText("next ›"));
    await waitFor(() => expect(mocks.page).toHaveBeenLastCalledWith("pdf1", 20, "stiffness"));
    fireEvent.click(screen.getByText("‹ previous"));
    await waitFor(() => expect(mocks.page).toHaveBeenLastCalledWith("pdf1", 12, "stiffness"));
  });

  it("says a scanned PDF cannot be searched instead of showing empty pages", () => {
    render(<PdfPages id="pdf1" doc={doc({ status: "no_text", text_pages: 0 })} />);
    expect(screen.getByText("CORTEX does not run OCR", { exact: false })).toBeTruthy();
    expect(screen.queryByTestId("page-text")).toBeNull();
    expect(mocks.page).not.toHaveBeenCalled();
  });
});
