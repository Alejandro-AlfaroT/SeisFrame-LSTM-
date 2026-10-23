import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, type SimulationLinkDatum, type SimulationNodeDatum } from "d3-force";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { Nav } from "../App";
import { errorText, useAsync } from "../lib";
import type { Category, GraphData } from "../types";
import { CATEGORY_REGION, REGION_META } from "../types";

type GNode = GraphData["nodes"][number] & SimulationNodeDatum;
type GLink = SimulationLinkDatum<GNode> & { id: string; rel_type: string; note: string | null };

const W = 400;
const H = 420;
const CATEGORY_SHADE: Record<Category, string> = { facts: "#f2f2f2", sessions: "#9aa0a6", inbox: "#6b7075", projects: "#d9e6c4" };

function layout(data: GraphData): { nodes: GNode[]; links: GLink[] } {
  const nodes: GNode[] = data.nodes.map((n) => ({ ...n }));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const links: GLink[] = data.edges.filter((e) => byId.has(e.source) && byId.has(e.target)).map((e) => ({ ...e, source: byId.get(e.source)!, target: byId.get(e.target)! }));
  const sim = forceSimulation(nodes)
    .force("link", forceLink<GNode, GLink>(links).distance(46).strength(0.6))
    .force("charge", forceManyBody().strength(-90))
    .force("center", forceCenter(W / 2, H / 2))
    .force("collide", forceCollide(9))
    .stop();
  for (let i = 0; i < 260; i++) sim.tick();
  // Fit the settled layout into the view, leaving room on the right for labels.
  const xs = nodes.map((n) => n.x ?? 0);
  const ys = nodes.map((n) => n.y ?? 0);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const scale = Math.min((W - 150) / Math.max(x1 - x0, 1), (H - 40) / Math.max(y1 - y0, 1), 3);
  for (const n of nodes) {
    n.x = 20 + ((n.x ?? 0) - x0) * scale + (W - 150 - (x1 - x0) * scale) / 2;
    n.y = 20 + ((n.y ?? 0) - y0) * scale + (H - 40 - (y1 - y0) * scale) / 2;
  }
  return { nodes, links };
}

export default function GraphPanel({ recordId, rev, nav }: { recordId: string | null; rev: number; nav: Nav }) {
  const [root, setRoot] = useState<string | null>(recordId);
  const [depth, setDepth] = useState(2);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const graph = useAsync(() => api.graph(root, depth, root ? 120 : 160), [root, depth, rev]);
  const types = useAsync(() => api.relationshipTypes(), []);
  useEffect(() => setRoot(recordId), [recordId]);
  const { nodes, links } = useMemo(() => (graph.data ? layout(graph.data) : { nodes: [], links: [] }), [graph.data]);
  const edge = links.find((l) => l.id === selectedEdge);
  const neighbours = useMemo(() => {
    const ids = new Set<string>();
    for (const l of links) {
      const s = (l.source as GNode).id;
      const t = (l.target as GNode).id;
      if (s === root) ids.add(t);
      if (t === root) ids.add(s);
    }
    return ids;
  }, [links, root]);

  const edit = async (fn: () => Promise<unknown>) => {
    setErr(null);
    try {
      await fn();
      setSelectedEdge(null);
      nav.changed();
    } catch (e) {
      setErr(errorText(e));
    }
  };

  return (
    <div className="graph">
      <h3 className="panel-title">Knowledge graph</h3>
      <div className="toolbar">
        <div className="seg">
          {[1, 2, 3].map((d) => (
            <button key={d} className={depth === d ? "active" : ""} disabled={!root} onClick={() => setDepth(d)}>
              depth {d}
            </button>
          ))}
        </div>
        <button className={root ? "" : "active"} onClick={() => setRoot(null)}>
          best-connected
        </button>
      </div>
      <p className="mono tiny muted">
        {root ? "neighbourhood of the selected record" : "most connected records"} · {nodes.length} nodes · {links.length} links · click a node to recentre, double-click to open, click a link to
        edit
      </p>
      {graph.error && <p className="notice warn">{graph.error}</p>}
      <svg viewBox={`0 0 ${W} ${H}`} className="graph-svg" role="img" aria-label="Relationship graph">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,0.35)" />
          </marker>
        </defs>
        {links.map((l) => {
          const s = l.source as GNode;
          const t = l.target as GNode;
          const active = l.id === selectedEdge || hover === s.id || hover === t.id;
          return (
            <g key={l.id} className="edge" onClick={() => setSelectedEdge(l.id)}>
              <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke="transparent" strokeWidth={8} />
              <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke={active ? "#c6ff3d" : "rgba(255,255,255,0.18)"} strokeWidth={active ? 1.4 : 0.8} markerEnd="url(#arrow)" />
              {active && (
                <text x={((s.x ?? 0) + (t.x ?? 0)) / 2} y={((s.y ?? 0) + (t.y ?? 0)) / 2 - 3} className="edge-label">
                  {l.rel_type}
                </text>
              )}
            </g>
          );
        })}
        {nodes.map((n) => {
          const isRoot = n.id === root;
          return (
            <g
              key={n.id}
              transform={`translate(${n.x},${n.y})`}
              className="node"
              onMouseEnter={() => setHover(n.id)}
              onMouseLeave={() => setHover(null)}
              onClick={() => setRoot(n.id)}
              onDoubleClick={() => nav.openRecord(n.id)}
            >
              <circle r={isRoot ? 7 : 3.5 + Math.min(4, Math.sqrt(n.degree))} fill={isRoot ? "#c6ff3d" : CATEGORY_SHADE[n.category]} opacity={n.superseded ? 0.4 : 0.95} />
              {(isRoot || hover === n.id || (neighbours.has(n.id) && neighbours.size <= 12) || nodes.length <= 8) && (
                <text x={9} y={3} className={`node-label${isRoot ? " lime" : ""}`}>
                  {n.title.length > 38 ? `${n.title.slice(0, 37)}…` : n.title}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="legend mono tiny">
        {(Object.keys(CATEGORY_SHADE) as Category[]).map((c) => (
          <span key={c}>
            <i style={{ background: CATEGORY_SHADE[c] }} /> {REGION_META[CATEGORY_REGION[c]].label}
          </span>
        ))}
      </div>
      {edge && (
        <div className="subpanel">
          <p className="tiny">
            <b>{(edge.source as GNode).title}</b> <span className="mono lime">{edge.rel_type}</span> <b>{(edge.target as GNode).title}</b>
          </p>
          <div className="toolbar">
            <select value={edge.rel_type} onChange={(e) => edit(() => api.updateRelationship(edge.id, { rel_type: e.target.value }))} aria-label="Relationship type">
              {(types.data ?? []).map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
            <button className="danger" onClick={() => window.confirm("Remove this relationship?") && edit(() => api.deleteRelationship(edge.id))}>
              Remove link
            </button>
            <button onClick={() => setSelectedEdge(null)}>Close</button>
          </div>
          {err && <p className="notice warn">{err}</p>}
        </div>
      )}
    </div>
  );
}
