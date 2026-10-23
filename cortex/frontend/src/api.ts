import type {
  Activity,
  BackupInfo,
  BrainMap,
  Category,
  ConflictDetail,
  ConflictSummary,
  GraphData,
  ImportReport,
  MemoryRecord,
  ProjectSummary,
  RecordSummary,
  SearchResult,
  Stats,
  Version,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Params = Record<string, string | number | boolean | string[] | null | undefined>;

export function buildUrl(path: string, params?: Params): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) value.forEach((v) => search.append(key, v));
    else search.append(key, String(value));
  }
  const qs = search.toString();
  return `/api${path}${qs ? `?${qs}` : ""}`;
}

function describe(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail))
    return detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? `${(d as { loc?: unknown[] }).loc?.slice(1).join(".") ?? ""}: ${(d as { msg: string }).msg}` : String(d)))
      .join("; ");
  return fallback;
}

export async function request<T>(method: string, path: string, body?: unknown, params?: Params): Promise<T> {
  const headers: Record<string, string> = { "X-Cortex-Client": "1" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(buildUrl(path, params), {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = (await res.json()).detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(res.status, describe(detail, `${res.status} ${res.statusText}`));
  }
  return (await res.json()) as T;
}

export interface SearchParams {
  q?: string;
  mode?: "fulltext" | "similar";
  category?: string[];
  kind?: string[];
  tag?: string[];
  project_id?: string;
  date_from?: string;
  date_to?: string;
  verified?: boolean;
  include_demo?: boolean;
  status?: string;
  limit?: number;
  offset?: number;
}

export interface NewRecord {
  title: string;
  category: Category;
  content?: string;
  kind?: string;
  status?: string | null;
  project_id?: string | null;
  source_ref?: string | null;
  confidence?: number | null;
  verified?: boolean;
  subject?: string | null;
  predicate?: string | null;
  value?: string | null;
  occurred_at?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
  links?: { target_id: string; rel_type: string; direction?: "outgoing" | "incoming" }[];
}

export type RecordPatch = Partial<Omit<NewRecord, "links">> & { expected_updated_at?: string };

export const api = {
  health: () => request<{ status: string; version: string; schema_version: number; database: string }>("GET", "/health"),
  stats: () => request<Stats>("GET", "/stats"),
  activity: (limit = 30) => request<Activity[]>("GET", "/activity", undefined, { limit }),
  brain: (includeDemo = true) => request<BrainMap>("GET", "/brain", undefined, { include_demo: includeDemo }),
  tags: () => request<{ tag: string; count: number }[]>("GET", "/tags"),
  projects: () => request<ProjectSummary[]>("GET", "/projects"),
  relationshipTypes: () => request<string[]>("GET", "/relationship-types"),

  search: (p: SearchParams) => request<SearchResult>("GET", "/search", undefined, { ...p }),
  resolve: (ref: string) => request<RecordSummary>("GET", "/resolve", undefined, { ref }),

  record: (id: string) => request<MemoryRecord>("GET", `/records/${id}`),
  createRecord: (body: NewRecord) => request<MemoryRecord>("POST", "/records", body),
  updateRecord: (id: string, patch: RecordPatch) => request<MemoryRecord>("PATCH", `/records/${id}`, patch),
  trashRecord: (id: string) => request<MemoryRecord>("DELETE", `/records/${id}`),
  restoreRecord: (id: string) => request<MemoryRecord>("POST", `/records/${id}/restore`),
  purgeRecord: (id: string) => request<{ purged: string }>("DELETE", `/records/${id}/purge`, undefined, { confirm: id }),
  versions: (id: string) => request<Version[]>("GET", `/records/${id}/versions`),
  revert: (id: string, versionId: number) => request<MemoryRecord>("POST", `/records/${id}/revert`, { version_id: versionId }),
  similar: (id: string, limit = 8) => request<SearchResult>("GET", `/records/${id}/similar`, undefined, { limit }),
  trash: () => request<RecordSummary[]>("GET", "/trash"),

  createRelationship: (source_id: string, target_id: string, rel_type: string, note?: string) =>
    request<{ id: string }>("POST", "/relationships", { source_id, target_id, rel_type, note }),
  updateRelationship: (id: string, patch: { rel_type?: string; note?: string }) =>
    request<{ id: string }>("PATCH", `/relationships/${id}`, patch),
  deleteRelationship: (id: string) => request<{ deleted: string }>("DELETE", `/relationships/${id}`),
  graph: (recordId: string | null, depth = 2, limit = 150) =>
    request<GraphData>("GET", "/graph", undefined, { record_id: recordId, depth, limit }),

  conflicts: (status?: string) => request<ConflictSummary[]>("GET", "/conflicts", undefined, { status }),
  conflict: (id: string) => request<ConflictDetail>("GET", `/conflicts/${id}`),
  scanConflicts: () =>
    request<{ scanned_records: number; statements: number; candidates: number; new_conflicts: string[] }>("POST", "/conflicts/scan"),
  flagConflict: (body: { record_a_id: string; record_b_id: string; excerpt_a: string; excerpt_b: string; explanation: string }) =>
    request<ConflictDetail>("POST", "/conflicts", body),
  resolveConflict: (id: string, resolution: string, note?: string) =>
    request<ConflictDetail>("POST", `/conflicts/${id}/resolve`, { resolution, note: note || null }),
  reopenConflict: (id: string) => request<ConflictDetail>("POST", `/conflicts/${id}/reopen`),

  importMarkdown: (body: {
    files: { path: string; content: string }[];
    route: "inbox" | "by_type";
    project_id?: string | null;
    source_label: string;
    dry_run: boolean;
    include_templates: boolean;
    update_edited: boolean;
  }) => request<ImportReport>("POST", "/import/markdown", body),
  imports: () => request<{ id: string; source: string; route: string; created_at: string; stats: string }[]>("GET", "/imports"),

  backup: (label = "manual") => request<BackupInfo & { created_at: string }>("POST", "/backup", { label }),
  backups: () => request<BackupInfo[]>("GET", "/backups"),
  loadDemo: () => request<{ created: number; conflicts?: number; already_loaded?: number }>("POST", "/demo"),
  purgeDemo: () => request<{ removed: number }>("DELETE", "/demo"),
};
