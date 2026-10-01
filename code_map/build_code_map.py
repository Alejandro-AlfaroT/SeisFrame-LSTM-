"""Code map of the repository: one node per Python script with what it does, one edge per script-to-script
connection with how the connection is made (2026-09-29).

Nodes carry the script's path, package, module name, the first paragraph of its docstring, its top-level
functions and classes, whether it is a command-line entry point or a test, its line count and its external
dependencies. Edges are found statically with the ast module: ``import``/``from ... import`` statements
resolved against the repository's package roots (top-level imports and imports inside functions are told
apart), the names each importer actually uses and calls, and subprocess or path references to another
script. Nothing is executed.

Scope: the scripts under "RC Structure" and "RC Hybrid Surrogate Model" (the active packages); pass
``--include`` with other folders, or with nothing, to map more of the repository.

Outputs (in this folder): code_map.json (the graph), CODE_MAP.md (readable index grouped by package) and
code_map.html (interactive view, cytoscape.js from cdnjs).

usage: python code_map/build_code_map.py [--root <repo root>] [--include "RC Structure" ...]
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "outputs", ".ipynb_checkpoints", "node_modules", "projector_share", "Results", "checkpoints", "checkpoints_sap2000"}
PACKAGE_ROOTS = ("RC Structure", "RC Hybrid Surrogate Model")       # importable roots besides the repository root
INCLUDE_DIRS = ("RC Structure", "RC Hybrid Surrogate Model")        # only scripts under these folders become nodes (--include overrides)
STDLIB_HINT = {"os", "sys", "json", "csv", "math", "time", "re", "ast", "argparse", "pathlib", "collections", "itertools", "functools",
               "dataclasses", "typing", "subprocess", "shutil", "tempfile", "unittest", "hashlib", "datetime", "copy", "random", "io",
               "logging", "warnings", "contextlib", "traceback", "importlib", "glob", "pickle", "statistics", "textwrap", "string",
               "types", "enum", "operator", "inspect", "struct", "zipfile", "multiprocessing", "concurrent", "queue", "threading",
               "signal", "socket", "platform", "abc", "numbers", "fractions", "decimal", "heapq", "bisect", "array", "gc", "pprint",
               "html", "urllib", "http", "email", "base64", "uuid", "secrets", "getpass", "locale", "codecs", "configparser", "shlex",
               "fnmatch", "stat", "errno", "select", "ctypes", "weakref", "atexit", "builtins", "__future__", "gzip", "tkinter", "sqlite3", "unittest.mock"}


def repo_root(start):
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    return start


def python_files(root, include=INCLUDE_DIRS):
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).parts
        if any(part in EXCLUDE_DIRS for part in rel):
            continue
        if include and rel[0] not in include:
            continue
        yield p


def first_paragraph(text):
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    para = []
    for ln in lines:
        if not ln.strip() and para:
            break
        if ln.strip():
            para.append(ln.strip())
    return " ".join(para)


def leading_comment(source):
    out = []
    for ln in source.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if body and not body.startswith("!") and "coding" not in body[:12]:
                out.append(body)
        elif s:
            break
    return " ".join(out)


class Analyzer(ast.NodeVisitor):
    """Collects imports (with their scope), attribute uses of import aliases, calls of imported names, and
    string literals that name other scripts."""

    def __init__(self):
        self.imports = []             # {module, names:[(name, alias)], level, lineno, lazy}
        self.alias_to_module = {}     # local alias -> imported module (import X as y)
        self.name_to_import = {}      # local name -> (module, original name) for from-imports
        self.attr_uses = defaultdict(lambda: defaultdict(int))     # alias -> attr -> count
        self.name_uses = defaultdict(int)                           # local from-imported name -> uses
        self.calls = defaultdict(int)                               # local name or alias.attr -> call count
        self.strings = []
        self.depth = 0
        self.defs = []
        self.classes = []
        self.has_main_guard = False
        self.uses_argparse = False

    def visit_FunctionDef(self, node):
        if self.depth == 0:
            self.defs.append(node.name)
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        if self.depth == 0:
            self.classes.append(node.name)
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_If(self, node):
        test = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
        if "__name__" in test and "__main__" in test:
            self.has_main_guard = True
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            self.imports.append({"module": a.name, "names": [], "level": 0, "lineno": node.lineno, "lazy": self.depth > 0})
            self.alias_to_module[a.asname or a.name.split(".")[0]] = a.name
            if a.name == "argparse":
                self.uses_argparse = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        names = [(a.name, a.asname or a.name) for a in node.names]
        self.imports.append({"module": mod, "names": names, "level": node.level, "lineno": node.lineno, "lazy": self.depth > 0})
        for name, local in names:
            self.name_to_import[local] = (mod, name, node.level)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and (node.value.id in self.alias_to_module or node.value.id in self.name_to_import):
            self.attr_uses[node.value.id][node.attr] += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.name_to_import:
            self.name_uses[node.id] += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id in self.name_to_import:
            self.calls[f.id] += 1
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id in self.alias_to_module or f.value.id in self.name_to_import):
            self.calls[f"{f.value.id}.{f.attr}"] += 1
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.endswith(".py") and len(node.value) < 200:
            self.strings.append(node.value)
        self.generic_visit(node)


def build(root, include=INCLUDE_DIRS):
    root = Path(root).resolve()
    files = list(python_files(root, include))
    by_rel = {p.relative_to(root).as_posix(): p for p in files}
    roots = [root] + [root / r for r in PACKAGE_ROOTS if (root / r).exists()]

    def module_of(path):
        for r in roots[1:] + [root]:
            try:
                rel = path.relative_to(r)
                return rel.with_suffix("").as_posix().replace("/", "."), r.relative_to(root).as_posix() or "."
            except ValueError:
                continue
        return path.stem, "."

    def resolve(module, level, importer):
        """Repository file for a dotted module, trying the importer's directory, its package root, then every root."""
        if level:
            base = importer.parent
            for _ in range(level - 1):
                base = base.parent
            cands = [base]
        else:
            cands = [importer.parent]
            for r in roots[1:]:
                if str(importer).startswith(str(r)):
                    cands.append(r)
            cands += roots
        parts = module.split(".") if module else []
        for base in cands:
            for k in range(len(parts), 0 if level else 0, -1):
                sub = base.joinpath(*parts[:k]) if k else base
                for cand in (sub.with_suffix(".py"), sub / "__init__.py"):
                    if cand.exists() and cand.resolve() in {p.resolve() for p in files}:
                        return cand.resolve()
            if level and parts == []:
                cand = base / "__init__.py"
                if cand.exists():
                    return cand.resolve()
        return None

    nodes, edges, unresolved = {}, [], defaultdict(list)
    analyses = {}
    file_set = {p.resolve(): p.relative_to(root).as_posix() for p in files}
    for p in files:
        rel = p.relative_to(root).as_posix()
        source = p.read_text(encoding="utf-8", errors="replace")
        node = {"id": rel, "name": p.name, "package": rel.split("/")[0] if "/" in rel else ".",
                "folder": p.parent.relative_to(root).as_posix() or ".", "lines": source.count("\n") + 1}
        node["module"], node["root"] = module_of(p)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            node.update(summary=leading_comment(source) or "(no docstring; syntax error prevents analysis)", parse_error=str(exc),
                        functions=[], classes=[], is_cli=False, is_test=p.name.startswith("test_") or "/tests/" in rel, external=[])
            nodes[rel] = node
            continue
        an = Analyzer()
        an.visit(tree)
        analyses[rel] = an
        doc = ast.get_docstring(tree)
        summary, summary_source = first_paragraph(doc), "docstring"
        if not summary:
            summary, summary_source = leading_comment(source), "leading comment"
        if not summary:
            parts = []
            if an.classes:
                parts.append("defines the class" + ("es " if len(an.classes) > 1 else " ") + ", ".join(an.classes[:6]))
            if an.defs:
                parts.append("defines " + ", ".join(an.defs[:8]) + (" and more" if len(an.defs) > 8 else ""))
            ext_top = sorted({i["module"].split(".")[0] for i in an.imports if i["level"] == 0 and i["module"].split(".")[0] not in STDLIB_HINT
                              and resolve(i["module"], 0, p) is None})
            if ext_top:
                parts.append("built on " + ", ".join(ext_top[:6]))
            if an.has_main_guard or an.uses_argparse:
                parts.append("runs as a script")
            summary = ("No docstring; " + "; ".join(parts) + ".") if parts else "No docstring and no top-level definitions (procedural script)."
            summary_source = "derived from the code"
        node.update(summary=summary, summary_source=summary_source,
                    docstring_full=doc or "", functions=an.defs, classes=an.classes,
                    is_cli=bool(an.has_main_guard or an.uses_argparse), has_main_guard=an.has_main_guard,
                    is_test=p.name.startswith("test_") or "/tests/" in rel)
        ext = set()
        for imp in an.imports:
            if imp["level"] == 0 and resolve(imp["module"], 0, p) is None:
                top = imp["module"].split(".")[0]
                in_repo = any((r / top).is_dir() or (r / f"{top}.py").exists() for r in roots)
                if top and top not in STDLIB_HINT and not in_repo:
                    ext.add(top)
        node["external"] = sorted(ext)
        nodes[rel] = node

    for rel, an in analyses.items():
        p = by_rel[rel]
        grouped = {}
        for imp in an.imports:
            target = resolve(imp["module"], imp["level"], p)
            # `from PKG import mod`: names that are submodule files get their own edge, whether or not PKG has an __init__
            if imp["names"]:
                pkg_dirs = []
                for base in ([p.parent] + [r for r in roots[1:] if str(p).startswith(str(r))] + roots):
                    d = base.joinpath(*imp["module"].split(".")) if imp["module"] else base
                    if imp["level"]:
                        d = p.parent
                        for _ in range(imp["level"] - 1):
                            d = d.parent
                        d = d.joinpath(*imp["module"].split(".")) if imp["module"] else d
                    if d.is_dir():
                        pkg_dirs.append(d)
                        break
                remaining = []
                for name, local in imp["names"]:
                    sub = next((d / f"{name}.py" for d in pkg_dirs if (d / f"{name}.py").exists()), None)
                    if sub is not None and sub.resolve() in file_set and file_set[sub.resolve()] != rel:
                        trel = file_set[sub.resolve()]
                        key = (trel, "lazy_import" if imp["lazy"] else "import")
                        g = grouped.setdefault(key, {"source": rel, "target": trel, "kind": key[1], "lines": [], "module_as_written": f"{imp['module']}.{name}",
                                                      "names": {}, "attributes_used": {}, "calls": {}})
                        g["lines"].append(imp["lineno"])
                        for attr, n in an.attr_uses.get(local, {}).items():
                            g["attributes_used"][attr] = n
                        for k, n in an.calls.items():
                            if k.startswith(local + "."):
                                g["calls"][k.split(".", 1)[1]] = n
                    else:
                        remaining.append((name, local))
                if not remaining:
                    continue
                imp = {**imp, "names": remaining}
            if target is None:
                top = imp["module"].split(".")[0] if imp["module"] else ""
                if imp["level"] or (top and top not in STDLIB_HINT and (root / top).exists()):
                    unresolved[rel].append(imp["module"] or f"(relative level {imp['level']})")
                continue
            trel = file_set[target]
            if trel == rel:
                continue
            key = (trel, "lazy_import" if imp["lazy"] else "import")
            g = grouped.setdefault(key, {"source": rel, "target": trel, "kind": key[1], "lines": [], "module_as_written": imp["module"],
                                          "names": {}, "attributes_used": {}, "calls": {}})
            g["lines"].append(imp["lineno"])
            if imp["names"]:
                for name, local in imp["names"]:
                    g["names"][name] = an.name_uses.get(local, 0)
                    if an.calls.get(local):
                        g["calls"][name] = an.calls[local]
            else:
                alias = next((a for a, m in an.alias_to_module.items() if m == imp["module"]), imp["module"].split(".")[0])
                for attr, n in an.attr_uses.get(alias, {}).items():
                    g["attributes_used"][attr] = n
                for k, n in an.calls.items():
                    if k.startswith(alias + "."):
                        g["calls"][k.split(".", 1)[1]] = n
        for g in grouped.values():
            g["names"] = dict(sorted(g["names"].items(), key=lambda kv: -kv[1]))
            g["attributes_used"] = dict(sorted(g["attributes_used"].items(), key=lambda kv: -kv[1]))
            g["calls"] = dict(sorted(g["calls"].items(), key=lambda kv: -kv[1]))
            used = list(g["names"]) or list(g["attributes_used"])
            called = list(g["calls"])
            how = f"{'imports inside a function' if g['kind'] == 'lazy_import' else 'imports'} {nodes[g['target']]['module']}"
            if g["names"]:
                how += ": " + ", ".join(list(g["names"])[:8]) + (" ..." if len(g["names"]) > 8 else "")
            elif g["attributes_used"]:
                how += "; uses " + ", ".join(list(g["attributes_used"])[:8]) + (" ..." if len(g["attributes_used"]) > 8 else "")
            if called:
                how += "; calls " + ", ".join(f"{k} ({n}x)" for k, n in list(g["calls"].items())[:6])
            g["how"] = how
            g["used_count"] = len(used)
            edges.append(g)
        # subprocess / path references to other scripts
        for s in an.strings:
            name = Path(s.replace("\\", "/")).name
            matches = [t for t in nodes if t.endswith("/" + name) or t == name]
            if len(matches) == 1 and matches[0] != rel:
                edges.append({"source": rel, "target": matches[0], "kind": "script_reference", "lines": [], "names": {}, "attributes_used": {},
                              "calls": {}, "module_as_written": s, "how": f"refers to the script file {s!r} (subprocess, path or documentation)",
                              "used_count": 0})
    incoming = defaultdict(int); outgoing = defaultdict(int)
    for e in edges:
        incoming[e["target"]] += 1; outgoing[e["source"]] += 1
    for rel, node in nodes.items():
        node["imported_by"] = incoming[rel]; node["imports_count"] = outgoing[rel]
        node["unresolved_imports"] = sorted(set(unresolved.get(rel, [])))
    graph = {"generated_by": "code_map/build_code_map.py", "repository": root.name, "include": list(include) if include else ["(whole repository)"],
             "roots": [r.relative_to(root).as_posix() or "." for r in roots],
             "counts": {"nodes": len(nodes), "edges": len(edges), "packages": len({n["package"] for n in nodes.values()})},
             "nodes": list(nodes.values()), "edges": edges}
    return graph


def write_markdown(graph, path):
    nodes = {n["id"]: n for n in graph["nodes"]}
    out_edges = defaultdict(list); in_edges = defaultdict(list)
    for e in graph["edges"]:
        out_edges[e["source"]].append(e); in_edges[e["target"]].append(e)
    L = ["# Code map", "", f"Repository `{graph['repository']}`, folders {', '.join('`' + f + '`' for f in graph['include'])}: {graph['counts']['nodes']} scripts, {graph['counts']['edges']} connections, "
         f"generated by `code_map/build_code_map.py` (static analysis; regenerate after code changes). Roots for imports: "
         + ", ".join(f"`{r}`" for r in graph["roots"]) + ".", ""]
    by_folder = defaultdict(list)
    for n in graph["nodes"]:
        by_folder[n["folder"]].append(n)
    for folder in sorted(by_folder, key=lambda f: (f == ".", f)):
        L.append(f"## {folder}")
        L.append("")
        for n in sorted(by_folder[folder], key=lambda n: n["name"]):
            flags = [f for f, v in (("cli", n.get("is_cli")), ("test", n.get("is_test")), ("parse error", n.get("parse_error"))) if v]
            L.append(f"### `{n['id']}`" + (f" ({', '.join(flags)})" if flags else ""))
            L.append("")
            L.append(n["summary"])
            L.append("")
            defs = n.get("classes", []) + n.get("functions", [])
            L.append(f"- {n['lines']} lines; " + (f"defines {', '.join(defs[:12])}" + (" ..." if len(defs) > 12 else "") if defs else "no top-level definitions")
                     + (f"; external: {', '.join(n['external'])}" if n.get("external") else "") + ".")
            if out_edges[n["id"]]:
                L.append("- Uses:")
                for e in sorted(out_edges[n["id"]], key=lambda e: e["target"]):
                    L.append(f"  - `{e['target']}`: {e['how']}")
            if in_edges[n["id"]]:
                users = sorted({e["source"] for e in in_edges[n["id"]]})
                L.append(f"- Used by ({len(users)}): " + ", ".join(f"`{u}`" for u in users[:15]) + (" ..." if len(users) > 15 else ""))
            if n.get("unresolved_imports"):
                L.append(f"- Unresolved repository imports: {', '.join(n['unresolved_imports'])}")
            L.append("")
    derived = [n for n in graph["nodes"] if n.get("summary_source") == "derived from the code"]
    if derived:
        L.append("## Scripts without a docstring (descriptions derived from the code)")
        L.append("")
        L.append(f"{len(derived)} scripts. Adding a module docstring to any of them replaces the derived description on the next run.")
        L.append("")
        for n in sorted(derived, key=lambda n: (-n["imported_by"], n["id"])):
            L.append(f"- `{n['id']}` (imported by {n['imported_by']}): {n['summary']}")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Code map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
<style>
:root{--bg:#ffffff;--fg:#1a1a1a;--panel:#f4f4f4;--line:#cccccc;--accent:#2b5797}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--fg);display:flex;height:100vh}
#cy{flex:1;min-width:0}
#side{width:380px;max-width:45vw;border-left:1px solid var(--line);background:var(--panel);padding:12px 16px;overflow:auto;font-size:13px}
#side h2{font-size:15px;margin:8px 0 4px}#side h3{font-size:13px;margin:12px 0 4px}
#controls{position:absolute;top:10px;left:10px;background:var(--panel);border:1px solid var(--line);padding:8px 10px;font-size:13px;border-radius:4px}
#controls input[type=text]{width:220px}
code{background:#e8e8e8;padding:0 3px;border-radius:2px;font-size:12px}
ul{padding-left:18px;margin:4px 0}li{margin:2px 0}
@media (max-width:700px){body{flex-direction:column}#side{width:auto;max-width:none;max-height:45vh;border-left:none;border-top:1px solid var(--line)}}
</style></head><body>
<div id="cy"></div>
<div id="controls">
  <input type="text" id="q" placeholder="find a script (path or name)"> <button id="fit">fit</button><br>
  <label><input type="checkbox" id="tests"> show tests</label>
  <label><input type="checkbox" id="lazy" checked> lazy imports</label>
  <label><input type="checkbox" id="refs" checked> script references</label>
  <span id="stats"></span>
</div>
<div id="side"><h2>Code map</h2><p>Click a node for what the script does and how it connects. Edge labels appear on hover. Colours: one per folder; node size grows with the number of scripts that import it.</p><div id="detail"></div></div>
<script>
const GRAPH = __GRAPH__;
const folders = [...new Set(GRAPH.nodes.map(n => n.folder))].sort();
const palette = ["#2b5797","#b9413a","#3a8a4a","#8a5a2b","#6b3e9c","#1f8a8a","#a2732b","#555555","#c2477e","#4b6a2b","#2d6f9f","#9c3e3e","#3a7a6a","#7a5a9c","#8a8a2b"];
const color = f => palette[folders.indexOf(f) % palette.length];
const els = [];
for (const n of GRAPH.nodes) els.push({data: {id: n.id, label: n.name, folder: n.folder, test: n.is_test, size: 18 + 4 * Math.sqrt(n.imported_by), node: n}});
GRAPH.edges.forEach((e, i) => els.push({data: {id: "e" + i, source: e.source, target: e.target, kind: e.kind, how: e.how, edge: e}}));
const cy = cytoscape({container: document.getElementById("cy"), elements: els,
  style: [
    {selector: "node", style: {"background-color": ele => color(ele.data("folder")), "label": "data(label)", "font-size": 9, "width": "data(size)", "height": "data(size)", "text-valign": "bottom", "text-margin-y": 3, "color": "#222"}},
    {selector: "edge", style: {"width": 1, "line-color": "#9a9a9a", "target-arrow-color": "#9a9a9a", "target-arrow-shape": "triangle", "curve-style": "bezier", "arrow-scale": 0.7, "opacity": 0.6}},
    {selector: "edge[kind = 'lazy_import']", style: {"line-style": "dashed"}},
    {selector: "edge[kind = 'script_reference']", style: {"line-style": "dotted", "line-color": "#c08a2b", "target-arrow-color": "#c08a2b"}},
    {selector: ".highlight", style: {"line-color": "#2b5797", "target-arrow-color": "#2b5797", "width": 2.5, "opacity": 1}},
    {selector: "node.selected", style: {"border-width": 3, "border-color": "#111"}},
    {selector: ".faded", style: {"opacity": 0.12}}
  ],
  layout: {name: "preset"}});
cy.layout({name: "cose", animate: false, nodeRepulsion: 12000, idealEdgeLength: 70, gravity: 0.6, numIter: 1500, componentSpacing: 60}).run();
cy.fit(cy.nodes().filter(n => n.degree() > 0 && !n.data("test")), 30);
function applyFilters() {
  const showTests = document.getElementById("tests").checked, showLazy = document.getElementById("lazy").checked, showRefs = document.getElementById("refs").checked;
  cy.batch(() => {
    cy.nodes().forEach(n => n.style("display", (!n.data("test") || showTests) ? "element" : "none"));
    cy.edges().forEach(e => {
      const k = e.data("kind"); const ok = (k !== "lazy_import" || showLazy) && (k !== "script_reference" || showRefs);
      e.style("display", ok && e.source().style("display") !== "none" && e.target().style("display") !== "none" ? "element" : "none");
    });
  });
  const vis = cy.nodes().filter(n => n.style("display") !== "none").length, ve = cy.edges().filter(e => e.style("display") !== "none").length;
  document.getElementById("stats").textContent = ` ${vis} scripts, ${ve} connections`;
}
["tests","lazy","refs"].forEach(id => document.getElementById(id).addEventListener("change", applyFilters));
applyFilters();
document.getElementById("fit").onclick = () => cy.fit(undefined, 30);
document.getElementById("q").addEventListener("input", ev => {
  const q = ev.target.value.trim().toLowerCase();
  cy.elements().removeClass("faded");
  if (!q) return;
  const hits = cy.nodes().filter(n => n.data("id").toLowerCase().includes(q));
  cy.elements().not(hits).not(hits.connectedEdges()).addClass("faded");
  if (hits.length) cy.fit(hits, 80);
});
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
cy.on("tap", "node", ev => {
  const n = ev.target.data("node");
  cy.elements().removeClass("highlight"); cy.nodes().removeClass("selected"); ev.target.addClass("selected");
  ev.target.connectedEdges().addClass("highlight");
  const out = GRAPH.edges.filter(e => e.source === n.id), inn = GRAPH.edges.filter(e => e.target === n.id);
  let h = `<h2><code>${esc(n.id)}</code></h2><p>${esc(n.summary)}</p>`;
  h += `<p>${n.lines} lines` + (n.is_cli ? ", command-line entry point" : "") + (n.is_test ? ", test" : "") + (n.external.length ? `; external: ${esc(n.external.join(", "))}` : "") + `</p>`;
  const defs = (n.classes || []).concat(n.functions || []);
  if (defs.length) h += `<h3>Defines</h3><p>${esc(defs.join(", "))}</p>`;
  h += `<h3>Uses (${out.length})</h3><ul>` + out.map(e => `<li><code>${esc(e.target)}</code>: ${esc(e.how)}</li>`).join("") + `</ul>`;
  h += `<h3>Used by (${inn.length})</h3><ul>` + inn.map(e => `<li><code>${esc(e.source)}</code>: ${esc(e.how)}</li>`).join("") + `</ul>`;
  if (n.unresolved_imports && n.unresolved_imports.length) h += `<p>Unresolved repository imports: ${esc(n.unresolved_imports.join(", "))}</p>`;
  document.getElementById("detail").innerHTML = h;
});
cy.on("mouseover", "edge", ev => { ev.target.style("label", ev.target.data("how")); ev.target.style({"font-size": 9, "text-rotation": "autorotate", "text-background-color": "#fff", "text-background-opacity": 0.9, "text-background-padding": 2}); });
cy.on("mouseout", "edge", ev => ev.target.style("label", ""));
</script></body></html>
"""


def write_html(graph, path):
    slim = {"nodes": [{k: v for k, v in n.items() if k != "docstring_full"} for n in graph["nodes"]], "edges": graph["edges"]}
    path.write_text(HTML.replace("__GRAPH__", json.dumps(slim).replace("</", "<\\/")), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="repository root (default: the git root above this file)")
    parser.add_argument("--out", default=None, help="output folder (default: this folder)")
    parser.add_argument("--include", nargs="*", default=list(INCLUDE_DIRS), help="top-level folders whose scripts become nodes (empty = whole repository)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root(HERE)
    out = Path(args.out).resolve() if args.out else HERE
    out.mkdir(parents=True, exist_ok=True)
    graph = build(root, tuple(args.include) or None)
    (out / "code_map.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
    write_markdown(graph, out / "CODE_MAP.md")
    write_html(graph, out / "code_map.html")
    unresolved = sum(len(n["unresolved_imports"]) for n in graph["nodes"])
    kinds = defaultdict(int)
    for e in graph["edges"]:
        kinds[e["kind"]] += 1
    print(f"{graph['counts']['nodes']} scripts, {graph['counts']['edges']} connections {dict(kinds)}, {unresolved} unresolved repository imports; "
          f"wrote {out / 'code_map.json'}, CODE_MAP.md, code_map.html")


if __name__ == "__main__":
    main()
