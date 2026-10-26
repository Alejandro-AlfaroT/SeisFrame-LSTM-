export type Category = "facts" | "sessions" | "inbox" | "projects";
export type RegionId = "prefrontal" | "frontal" | "temporal" | "hippocampus" | "occipital";

export interface RecordSummary {
  id: string;
  title: string;
  category: Category;
  region: RegionId;
  kind: string;
  status: string | null;
  project_id: string | null;
  occurred_at: string | null;
  created_at: string;
  updated_at: string;
  verified: boolean;
  is_demo: boolean;
  deleted: boolean;
  tags: string[];
  excerpt: string;
  suggested_category: Category | null;
  superseded_by: string | null;
  snippet?: string | null;
  score?: number | null;
  deleted_at?: string;
}

export interface RelationshipView {
  id: string;
  rel_type: string;
  note: string | null;
  origin: string;
  other: { id: string; title: string; category: Category; kind: string; deleted: boolean };
}

export interface MemoryRecord {
  id: string;
  title: string;
  content: string;
  category: Category;
  region: RegionId;
  kind: string;
  status: string | null;
  project_id: string | null;
  project: { id: string; title: string } | null;
  source_ref: string | null;
  source_hash: string | null;
  confidence: number | null;
  verified: boolean;
  verified_at: string | null;
  subject: string | null;
  predicate: string | null;
  value: string | null;
  occurred_at: string | null;
  metadata: Record<string, unknown>;
  is_demo: boolean;
  deleted: boolean;
  deleted_at: string | null;
  created_at: string;
  updated_at: string;
  tags: string[];
  relationships: { outgoing: RelationshipView[]; incoming: RelationshipView[] };
  version_count: number;
  conflicts: { id: string; status: string; reason: string; record_a_id: string; record_b_id: string }[];
  document?: DocumentInfo | null;
}

/** What was read from a PDF record's file. */
export interface DocumentInfo {
  path: string;
  sha256: string;
  bytes: number;
  page_count: number;
  text_pages: number;
  failed_pages: number;
  status: "ok" | "partial" | "no_text" | "failed";
  error: string | null;
  pdf_meta: Record<string, string>;
  extractor: string;
  extracted_at: string;
}

export interface DocumentHit extends RecordSummary {
  page_count: number;
  pages_matched: number;
  passages: { page: number; snippet: string; score: number }[];
}

export interface DocumentPage {
  record_id: string;
  page: number;
  page_count: number;
  has_text: boolean;
  text: string;
  matches: number[];
}

export interface Stats {
  total_records: number;
  by_category: Record<Category, number>;
  verified_facts: number;
  unverified_facts: number;
  active_projects: number;
  sessions: number;
  inbox: number;
  open_conflicts: number;
  relationships: number;
  pdfs: number;
  pdf_pages_searchable: number;
  trash: number;
  demo_records: number;
  last_backup: string | null;
}

export interface Activity {
  id: number;
  at: string;
  action: string;
  record_id: string | null;
  title: string | null;
  detail: string | null;
}

export interface Cluster {
  id: string;
  region: RegionId;
  label: string;
  count: number;
  record_ids: string[];
  conflict_ids: string[];
}

export interface BrainRegion {
  region: RegionId;
  category: Category | "conflicts";
  label: string;
  count: number;
  clusters: Cluster[];
}

export interface BrainNode {
  id: string;
  title: string;
  category: Category;
  kind: string;
  region: RegionId;
  cluster_id: string;
  verified: boolean;
  is_demo: boolean;
}

export interface BrainEdge {
  id: string;
  source: string;
  target: string;
  rel_type: string;
}

export interface BrainMap {
  regions: BrainRegion[];
  nodes: BrainNode[];
  edges: BrainEdge[];
}

export interface SearchResult {
  query?: string;
  mode: "browse" | "fulltext" | "similar";
  method?: string;
  total: number;
  results: RecordSummary[];
  error?: string;
  documents?: {
    total_pages: number;
    total_documents: number;
    capped: boolean;
    results: DocumentHit[];
    error?: string;
  } | null;
}

export interface ConflictSummary {
  id: string;
  status: "open" | "resolved" | "dismissed";
  reason: string;
  explanation: string;
  score: number;
  excerpt_a: string;
  excerpt_b: string;
  resolution: string | null;
  resolution_note: string | null;
  detected_at: string;
  resolved_at: string | null;
  is_demo: boolean;
  a: { id: string; title: string; category: Category; deleted: boolean };
  b: { id: string; title: string; category: Category; deleted: boolean };
}

export interface ConflictDetail {
  id: string;
  status: "open" | "resolved" | "dismissed";
  reason: string;
  explanation: string;
  score: number;
  excerpt_a: string;
  excerpt_b: string;
  resolution: string | null;
  resolution_note: string | null;
  detected_at: string;
  resolved_at: string | null;
  record_a: MemoryRecord;
  record_b: MemoryRecord;
}

export interface GraphData {
  root: string | null;
  nodes: {
    id: string;
    title: string;
    category: Category;
    kind: string;
    status: string | null;
    verified: boolean;
    is_demo: boolean;
    superseded: boolean;
    degree: number;
  }[];
  edges: { id: string; source: string; target: string; rel_type: string; note: string | null }[];
}

export interface ProjectSummary extends RecordSummary {
  objectives: string[];
  record_counts: Record<string, number>;
  milestones: { id: string; title: string; status: string | null; occurred_at: string | null }[];
}

export interface ImportReport {
  batch_id: string | null;
  dry_run: boolean;
  route: string;
  source: string;
  files: { path: string; action: string; title: string; category: string; kind: string; warnings: string[]; record_id?: string }[];
  skipped: { path: string; reason: string }[];
  counts: Record<string, number>;
  links: { resolved: number; unresolved: number };
  conflicts: { new: number } | null;
}

export interface Version {
  id: number;
  reason: string;
  created_at: string;
  snapshot: Record<string, unknown>;
}

export interface BackupInfo {
  name: string;
  path: string;
  bytes: number;
}

export const REGION_META: Record<RegionId, { label: string; category: Category | "conflicts"; anatomy: string }> = {
  prefrontal: { label: "Projects", category: "projects", anatomy: "PREFRONTAL" },
  frontal: { label: "Conflicts", category: "conflicts", anatomy: "FRONTAL" },
  temporal: { label: "Inbox", category: "inbox", anatomy: "TEMPORAL" },
  hippocampus: { label: "Facts", category: "facts", anatomy: "HIPPOCAMPUS" },
  occipital: { label: "Sessions", category: "sessions", anatomy: "OCCIPITAL" },
};

export const CATEGORY_REGION: Record<Category, RegionId> = {
  projects: "prefrontal",
  inbox: "temporal",
  facts: "hippocampus",
  sessions: "occipital",
};

export const REGION_ORDER: RegionId[] = ["prefrontal", "frontal", "temporal", "hippocampus", "occipital"];
