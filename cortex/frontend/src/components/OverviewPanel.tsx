import type { Nav } from "../App";
import type { Stats } from "../types";
import { REGION_META, REGION_ORDER } from "../types";

export default function OverviewPanel({ stats, nav }: { stats?: Stats; nav: Nav }) {
  const empty = stats && stats.total_records === 0;
  return (
    <div className="overview">
      <h3 className="panel-title">CORTEX</h3>
      <p>
        A local knowledge base for the SeisFrame project. Notes, facts, decisions and work sessions live in a SQLite file on this machine. Nothing is sent to external services.
      </p>
      {empty && (
        <div className="notice">
          The database is empty.{" "}
          <button className="link" onClick={() => nav.go({ kind: "import" })}>
            Import Markdown notes
          </button>
          ,{" "}
          <button className="link" onClick={() => nav.go({ kind: "capture" })}>
            capture something
          </button>
          , or{" "}
          <button className="link" onClick={() => nav.go({ kind: "data" })}>
            load the demo data
          </button>
          .
        </div>
      )}
      <h4 className="section-title">How the brain maps to memory</h4>
      <ul className="plain legend-list">
        {REGION_ORDER.map((r) => (
          <li key={r}>
            <button className="link" onClick={() => nav.go({ kind: "region", region: r })}>
              <span className="mono lime">{REGION_META[r].anatomy}</span> → {REGION_META[r].label}
            </button>
          </li>
        ))}
      </ul>
      <p className="tiny muted">The mapping is an organizational metaphor, not a description of brain function.</p>
      <h4 className="section-title">Using it</h4>
      <ul className="tips">
        <li>Click a region on the brain or in the list to see its clusters. Click a cluster to list its records, or a bright node to open one.</li>
        <li>Lines inside the brain are typed relationships. Opening a record lights up its neighbours.</li>
        <li>AXIAL / SAGITTAL / CORONAL change the camera. The three thumbnails are flat projections of the same particles.</li>
        <li>Search lights up matching records. Use "quotes" for phrases and -word to exclude.</li>
        <li>Possible contradictions show up in Conflicts. Only you resolve them, and nothing is deleted.</li>
      </ul>
    </div>
  );
}
