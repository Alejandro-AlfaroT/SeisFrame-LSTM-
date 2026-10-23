import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { relativePath } from "./components/ImportPanel";
import { ago, fmtDate, Snippet, SNIPPET_CLOSE, SNIPPET_OPEN } from "./lib";

afterEach(cleanup);

describe("Snippet", () => {
  it("highlights FTS matches and never interprets HTML", () => {
    const { container } = render(<Snippet text={`<b>x</b> theta_u is measured from the ${SNIPPET_OPEN}origin${SNIPPET_CLOSE}.`} />);
    expect(container.querySelector("mark")?.textContent).toBe("origin");
    expect(container.querySelector("b")).toBeNull();
    expect(container.textContent).toContain("<b>x</b>");
  });
});

describe("dates", () => {
  it("formats ISO dates and relative times", () => {
    expect(fmtDate("2026-10-22")).toBe("2026-10-22");
    expect(fmtDate(null)).toBe("—");
    const now = Date.parse("2026-10-22T12:00:00Z");
    expect(ago("2026-10-22T11:59:30Z", now)).toBe("30s");
    expect(ago("2026-10-22T09:00:00Z", now)).toBe("3h");
    expect(ago("2026-10-20T12:00:00Z", now)).toBe("2d");
  });
});

describe("relativePath", () => {
  it("keeps sub-folders of a folder upload and labels the source by its root", () => {
    const file = { name: "DEC-001.md", webkitRelativePath: "SeisFrame-Obsidian-Starter/06 Decisions/DEC-001.md" } as File;
    expect(relativePath(file)).toEqual({ label: "SeisFrame-Obsidian-Starter", path: "06 Decisions/DEC-001.md" });
    expect(relativePath({ name: "note.md", webkitRelativePath: "" } as File)).toEqual({ label: "upload", path: "note.md" });
  });
});
