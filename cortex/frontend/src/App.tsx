import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from "react";

import { api } from "./api";
import { generateBrain, placeClusters, type Placement } from "./brain/brainModel";
import type { ViewPreset } from "./brain/BrainView";
import SliceViews from "./brain/SliceViews";
import CapturePanel, { type CaptureMode } from "./components/CapturePanel";
import ConflictPanel from "./components/ConflictPanel";
import { ActivityFeed, Dashboard, RegionNav } from "./components/Dashboard";
import DataPanel from "./components/DataPanel";
import GraphPanel from "./components/GraphPanel";
import ImportPanel from "./components/ImportPanel";
import OverviewPanel from "./components/OverviewPanel";
import RecordPanel from "./components/RecordPanel";
import { ClusterPanel, RegionPanel } from "./components/RegionPanel";
import SearchPanel from "./components/SearchPanel";
import { useAsync } from "./lib";
import type { Cluster, RegionId } from "./types";
import { REGION_META } from "./types";

const BrainView = lazy(() => import("./brain/BrainView"));

export type Panel =
  | { kind: "overview" }
  | { kind: "region"; region: RegionId }
  | { kind: "cluster"; clusterId: string }
  | { kind: "record"; id: string; page?: number; q?: string }
  | { kind: "search"; q?: string }
  | { kind: "capture"; mode?: CaptureMode }
  | { kind: "import" }
  | { kind: "conflicts"; id?: string }
  | { kind: "graph"; id?: string | null }
  | { kind: "data" };

export interface Nav {
  go: (p: Panel) => void;
  /** `at` opens a PDF record at a page, with the words of `q` marked. */
  openRecord: (id: string, at?: { page: number; q?: string }) => void;
  openRef: (ref: string) => void;
  changed: () => void;
  toast: (msg: string) => void;
}

const EMPTY_PLACEMENT: Placement = { anchors: new Map(), nodes: new Map() };

export default function App() {
  const [rev, setRev] = useState(0);
  const changed = useCallback(() => setRev((r) => r + 1), []);
  const [includeDemo, setIncludeDemo] = useState(true);
  const stats = useAsync(() => api.stats(), [rev]);
  const activity = useAsync(() => api.activity(16), [rev]);
  const brain = useAsync(() => api.brain(includeDemo), [rev, includeDemo]);

  const cloud = useMemo(() => generateBrain(), []);
  const clusters = useMemo(() => brain.data?.regions.flatMap((r) => r.clusters) ?? [], [brain.data]);
  const membersOf = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const n of brain.data?.nodes ?? []) {
      if (!m.has(n.cluster_id)) m.set(n.cluster_id, []);
      m.get(n.cluster_id)!.push(n.id);
    }
    return m;
  }, [brain.data]);
  const placement = useMemo(
    () => (brain.data ? placeClusters(cloud, clusters, (id) => membersOf.get(id) ?? []) : EMPTY_PLACEMENT),
    [cloud, brain.data, clusters, membersOf],
  );

  const [panel, setPanel] = useState<Panel>({ kind: "overview" });
  const [history, setHistory] = useState<Panel[]>([]);
  const [view, setView] = useState<ViewPreset>("3d");
  const [viewNonce, setViewNonce] = useState(0);
  const [hoverRegion, setHoverRegion] = useState<RegionId | null>(null);
  const [searchHits, setSearchHits] = useState<string[]>([]);
  const [conflictFocus, setConflictFocus] = useState<string[]>([]);
  const [toastText, setToastText] = useState<string | null>(null);

  const toast = useCallback((msg: string) => {
    setToastText(msg);
    window.setTimeout(() => setToastText((t) => (t === msg ? null : t)), 4200);
  }, []);
  const go = useCallback(
    (p: Panel) => {
      setHistory((h) => [...h.slice(-40), panel]);
      setPanel(p);
    },
    [panel],
  );
  const back = () => {
    setHistory((h) => {
      const prev = h[h.length - 1];
      if (prev) setPanel(prev);
      return h.slice(0, -1);
    });
  };
  const openRecord = useCallback((id: string, at?: { page: number; q?: string }) => go({ kind: "record", id, ...at }), [go]);
  const openRef = useCallback(
    (ref: string) => {
      api
        .resolve(ref)
        .then((r) => go({ kind: "record", id: r.id }))
        .catch(() => toast(`No record matches “${ref}”. It may not have been imported.`));
    },
    [go, toast],
  );
  const nav: Nav = { go, openRecord, openRef, changed, toast };
  const setViewPreset = (v: ViewPreset) => {
    setView(v);
    setViewNonce((n) => n + 1);
  };

  const nodeById = useMemo(() => new Map((brain.data?.nodes ?? []).map((n) => [n.id, n])), [brain.data]);
  const cluster = panel.kind === "cluster" ? clusters.find((c) => c.id === panel.clusterId) : undefined;
  const recordId = panel.kind === "record" ? panel.id : null;

  const selectedRegion: RegionId | null =
    panel.kind === "region"
      ? panel.region
      : cluster
        ? cluster.region
        : panel.kind === "conflicts"
          ? "frontal"
          : recordId
            ? (nodeById.get(recordId)?.region ?? null)
            : null;

  const highlightIds = useMemo(() => {
    if (cluster) return new Set(cluster.record_ids);
    if (recordId) {
      const ids = new Set([recordId]);
      for (const e of brain.data?.edges ?? []) {
        if (e.source === recordId) ids.add(e.target);
        if (e.target === recordId) ids.add(e.source);
      }
      return ids;
    }
    if (panel.kind === "search") return new Set(searchHits);
    if (panel.kind === "conflicts") return new Set(conflictFocus);
    return new Set<string>();
  }, [cluster, recordId, brain.data, panel.kind, searchHits, conflictFocus]);

  useEffect(() => {
    if (panel.kind !== "search") setSearchHits([]);
    if (panel.kind !== "conflicts") setConflictFocus([]);
  }, [panel.kind]);

  const selectCluster = (c: Cluster) => {
    if (c.region === "frontal" && c.conflict_ids.length === 1) go({ kind: "conflicts", id: c.conflict_ids[0] });
    else go({ kind: "cluster", clusterId: c.id });
  };

  const renderPanel = () => {
    switch (panel.kind) {
      case "overview":
        return <OverviewPanel stats={stats.data} nav={nav} />;
      case "region":
        return <RegionPanel key={panel.region} region={panel.region} map={brain.data ?? null} rev={rev} nav={nav} onCluster={selectCluster} />;
      case "cluster":
        return cluster ? <ClusterPanel cluster={cluster} map={brain.data ?? null} nav={nav} /> : <p className="muted pad">Cluster no longer exists.</p>;
      case "record":
        return <RecordPanel key={`${panel.id}:${panel.page ?? ""}:${panel.q ?? ""}`} id={panel.id} rev={rev} nav={nav} page={panel.page} q={panel.q} />;
      case "search":
        return <SearchPanel initialQuery={panel.q ?? ""} nav={nav} onResults={setSearchHits} />;
      case "capture":
        return <CapturePanel initialMode={panel.mode ?? "note"} nav={nav} />;
      case "import":
        return <ImportPanel nav={nav} />;
      case "conflicts":
        return <ConflictPanel key={panel.id ?? "list"} conflictId={panel.id} rev={rev} nav={nav} onFocus={setConflictFocus} />;
      case "graph":
        return <GraphPanel recordId={panel.id ?? null} rev={rev} nav={nav} />;
      case "data":
        return <DataPanel rev={rev} nav={nav} includeDemo={includeDemo} setIncludeDemo={setIncludeDemo} />;
    }
  };

  const openConflicts = stats.data?.open_conflicts ?? 0;
  const hud = selectedRegion ? REGION_META[selectedRegion] : null;

  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={() => go({ kind: "overview" })} title="Overview">
          <span className="logo">CORTEX</span>
          <span className="sub">SeisFrame memory · local</span>
        </button>
        <form
          className="global-search"
          onSubmit={(e) => {
            e.preventDefault();
            const q = new FormData(e.currentTarget).get("q")?.toString() ?? "";
            go({ kind: "search", q });
          }}
        >
          <input name="q" placeholder="Search all memory  ⏎" aria-label="Search all memory" />
        </form>
        <nav className="actions">
          <button className="primary" onClick={() => go({ kind: "capture" })}>
            + Capture
          </button>
          <button onClick={() => go({ kind: "import" })}>Import</button>
          <button onClick={() => go({ kind: "conflicts" })}>
            Conflicts {openConflicts > 0 && <span className="badge">{openConflicts}</span>}
          </button>
          <button onClick={() => go({ kind: "graph", id: recordId })}>Graph</button>
          <button onClick={() => go({ kind: "data" })}>Data</button>
        </nav>
      </header>

      <aside className="left">
        <Dashboard stats={stats.data} error={stats.error} includeDemo={includeDemo} onToggleDemo={() => setIncludeDemo((v) => !v)} nav={nav} />
        <RegionNav map={brain.data ?? null} selected={selectedRegion} onHover={setHoverRegion} onSelect={(r) => go({ kind: "region", region: r })} />
        <ActivityFeed items={activity.data ?? []} nav={nav} />
      </aside>

      <main className="viewport">
        <Suspense fallback={<div className="loading">assembling particles…</div>}>
          <BrainView
            cloud={cloud}
            map={brain.data ?? null}
            placement={placement}
            view={view}
            viewNonce={viewNonce}
            selectedRegion={selectedRegion}
            hoverRegion={hoverRegion}
            selectedClusterId={cluster?.id ?? null}
            selectedRecordId={recordId}
            highlightIds={highlightIds}
            onHoverRegion={setHoverRegion}
            onSelectRegion={(r) => (r ? go({ kind: "region", region: r }) : panel.kind === "region" && go({ kind: "overview" }))}
            onSelectCluster={selectCluster}
            onSelectRecord={openRecord}
          />
        </Suspense>
        <div className="hud tl">
          {(["3d", "axial", "sagittal", "coronal"] as ViewPreset[]).map((v) => (
            <button key={v} className={view === v ? "active" : ""} onClick={() => setViewPreset(v)}>
              {v.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="hud tr mono">
          <div>
            VIEW <b>{view.toUpperCase()}</b>
          </div>
          <div>PARTICLES {cloud.count.toLocaleString()}</div>
          <div>
            NODES {brain.data?.nodes.length ?? 0} · LINKS {brain.data?.edges.length ?? 0}
          </div>
          {hud && (
            <div className="lime">
              ▸ {hud.anatomy} / {hud.label.toUpperCase()}
            </div>
          )}
          {!hud && hoverRegion && <div>◦ {REGION_META[hoverRegion].anatomy}</div>}
          {brain.error && <div className="warn">API: {brain.error}</div>}
          <div className="dimmer">region ≙ memory category · metaphor, not anatomy</div>
        </div>
        <SliceViews cloud={cloud} placement={placement} selectedRegion={selectedRegion} highlightIds={highlightIds} view={view} onView={setViewPreset} />
      </main>

      <section className="panel">
        <div className="panel-head">
          <button className="ghost" onClick={back} disabled={!history.length} title="Back">
            ←
          </button>
          <span className="mono muted crumbs">{panel.kind.toUpperCase()}</span>
        </div>
        <div className="panel-body">{renderPanel()}</div>
      </section>

      {toastText && (
        <div className="toast" role="status" onClick={() => setToastText(null)}>
          {toastText}
        </div>
      )}
    </div>
  );
}
