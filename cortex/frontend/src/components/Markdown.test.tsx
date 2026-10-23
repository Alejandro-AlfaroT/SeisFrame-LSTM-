import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Markdown, { REF_PREFIX, safeUrl, wikilinksToLinks } from "./Markdown";

afterEach(cleanup);

describe("wikilinksToLinks", () => {
  it("turns Obsidian links into record references, keeping aliases", () => {
    expect(wikilinksToLinks("See [[07 Daily/2026-10-22#^t01|T01]] and [[DEC-001]].")).toBe(
      `See [T01](${REF_PREFIX}07%20Daily%2F2026-10-22%23%5Et01) and [DEC-001](${REF_PREFIX}DEC-001).`,
    );
  });
});

describe("safeUrl", () => {
  it("allows web and mail links only", () => {
    expect(safeUrl("https://doi.org/10.1002/eqe.495")).toBe("https://doi.org/10.1002/eqe.495");
    expect(safeUrl("mailto:me@example.com")).toBe("mailto:me@example.com");
    expect(safeUrl("javascript:alert(1)")).toBe("");
    expect(safeUrl("JaVaScRiPt:alert(1)")).toBe("");
    expect(safeUrl("data:text/html,<script>x</script>")).toBe("");
    expect(safeUrl("file:///C:/secret.txt")).toBe("");
    expect(safeUrl("#heading")).toBe("");
  });

  it("treats relative paths as references to other notes", () => {
    expect(safeUrl("03 Literature/Ibarra 2005.md")).toBe(`${REF_PREFIX}03%20Literature%2FIbarra%202005.md`);
  });
});

describe("<Markdown>", () => {
  it("never renders raw HTML or loads images from untrusted notes", () => {
    const { container } = render(
      <Markdown
        text={'<script>window.pwned = 1</script>\n\n<img src="https://tracker.example/p.gif">\n\n![alt text](https://tracker.example/x.png)\n\n[click](javascript:alert(1))'}
        onOpenRef={() => {}}
      />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByText("[image not loaded: alt text]")).toBeTruthy();
    const link = screen.getByText("click");
    expect(link.closest("a")?.getAttribute("href") ?? "").not.toContain("javascript");
  });

  it("opens wikilinks inside the app and external links in a new tab", () => {
    const onOpenRef = vi.fn();
    render(<Markdown text={"Read [[03 Literature/Ibarra 2005|Ibarra]] or [the paper](https://doi.org/10.1002/eqe.495)."} onOpenRef={onOpenRef} />);
    fireEvent.click(screen.getByText("Ibarra"));
    expect(onOpenRef).toHaveBeenCalledWith("03 Literature/Ibarra 2005");
    const external = screen.getByText("the paper").closest("a")!;
    expect(external.getAttribute("target")).toBe("_blank");
    expect(external.getAttribute("rel")).toContain("noopener");
  });

  it("renders GFM tables", () => {
    const { container } = render(<Markdown text={"| ID | Status |\n| --- | --- |\n| DEC-001 | accepted |"} onOpenRef={() => {}} />);
    expect(container.querySelectorAll("td")).toHaveLength(2);
  });
});
