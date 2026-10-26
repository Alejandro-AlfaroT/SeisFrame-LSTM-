import type { Nav } from "../App";
import { ago } from "../lib";
import type { Activity, BrainMap, RegionId, Stats } from "../types";
import { REGION_META, REGION_ORDER } from "../types";

function Tile({ label, value, sub, tone, onClick }: { label: string; value: number | string; sub?: string; tone?: "lime" | "warn"; onClick?: () => void }) {
  return (
    <button className={`tile${tone ? ` ${tone}` : ""}`} onClick={onClick} disabled={!onClick}>
      <span className="tile-value">{value}</span>
      <span className="tile-label">{label}</span>
      {sub && <span className="tile-sub">{sub}</span>}
    </button>
  );
}

export function Dashboard({ stats, error, includeDemo, onToggleDemo, nav }: { stats?: Stats; error: string | null; includeDemo: boolean; onToggleDemo: () => void; nav: Nav }) {
  if (error) return <div className="block warn mono">Backend unreachable: {error}</div>;
  if (!stats) return <div className="block mono muted">loading…</div>;
  const facts = stats.verified_facts + stats.unverified_facts;
  return (
    <div className="block">
      <h3 className="block-title">Dashboard</h3>
      <div className="tiles">
        <Tile label="Total notes" value={stats.total_records} onClick={() => nav.go({ kind: "search" })} />
        <Tile label="Verified facts" value={stats.verified_facts} sub={`of ${facts}`} tone={stats.verified_facts ? "lime" : undefined} onClick={() => nav.go({ kind: "region", region: "hippocampus" })} />
        <Tile label="Active projects" value={stats.active_projects} onClick={() => nav.go({ kind: "region", region: "prefrontal" })} />
        <Tile label="Sessions" value={stats.sessions} onClick={() => nav.go({ kind: "region", region: "occipital" })} />
        <Tile label="Open conflicts" value={stats.open_conflicts} tone={stats.open_conflicts ? "warn" : undefined} onClick={() => nav.go({ kind: "conflicts" })} />
        <Tile label="Inbox" value={stats.inbox} onClick={() => nav.go({ kind: "region", region: "temporal" })} />
      </div>
      <div className="mono tiny muted meta-line">
        {stats.relationships} links · {stats.pdfs > 0 && `${stats.pdfs} PDFs (${stats.pdf_pages_searchable.toLocaleString()} pages searchable) · `}
        {stats.trash} in trash · backup {stats.last_backup ? stats.last_backup.slice(7, 22) : "never"}
      </div>
      {stats.demo_records > 0 && (
        <div className="demo-flag mono tiny">
          DEMO DATA · {stats.demo_records} records{" "}
          <button className="link" onClick={onToggleDemo}>
            {includeDemo ? "hide in brain" : "show in brain"}
          </button>
        </div>
      )}
    </div>
  );
}

export function RegionNav({ map, selected, onHover, onSelect }: { map: BrainMap | null; selected: RegionId | null; onHover: (r: RegionId | null) => void; onSelect: (r: RegionId) => void }) {
  return (
    <div className="block">
      <h3 className="block-title">Memory regions</h3>
      <ul className="region-nav">
        {REGION_ORDER.map((r) => {
          const region = map?.regions.find((x) => x.region === r);
          return (
            <li key={r}>
              <button className={selected === r ? "active" : ""} onMouseEnter={() => onHover(r)} onMouseLeave={() => onHover(null)} onClick={() => onSelect(r)}>
                <span className="mono anat">{REGION_META[r].anatomy}</span>
                <span className="cat">{REGION_META[r].label}</span>
                <span className="mono count">{region?.count ?? 0}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

const ACTION_LABEL: Record<string, string> = {
  created: "added",
  updated: "edited",
  trashed: "trashed",
  restored: "restored",
  purged: "deleted",
  linked: "linked",
  unlinked: "unlinked",
  relinked: "relinked",
  imported: "imported",
  backup: "backup",
  "conflicts-detected": "conflicts",
  "conflict-resolved": "resolved",
  "conflict-reopened": "reopened",
  "conflict-flagged": "flagged",
  "demo-loaded": "demo",
  "demo-removed": "demo",
};

export function ActivityFeed({ items, nav }: { items: Activity[]; nav: Nav }) {
  return (
    <div className="block grow">
      <h3 className="block-title">Recent activity</h3>
      {items.length === 0 && <p className="muted tiny">Nothing yet.</p>}
      <ul className="activity">
        {items.map((a) => (
          <li key={a.id}>
            <span className="mono when">{ago(a.at)}</span>
            <span className="mono what">{ACTION_LABEL[a.action] ?? a.action}</span>
            {a.record_id && a.action !== "purged" ? (
              <button className="link" onClick={() => nav.openRecord(a.record_id!)}>
                {a.title}
              </button>
            ) : (
              <span>{a.title}</span>
            )}
            {a.detail && <span className="muted detail">{a.detail}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
