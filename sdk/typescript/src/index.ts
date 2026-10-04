/**
 * llm-wiki TypeScript KnowledgeProvider SDK — agent protocol v1.
 *
 * A dependency-free fetch client over the frozen `/v1` surface documented in
 * `docs/agent-protocol.md`. Every response carries `protocol_version` (this
 * SDK asserts it) and `request_id` (surfaced on errors for support).
 *
 * Workspace isolation: one client targets one server process, and one server
 * process serves exactly one workspace — create one client per workspace.
 */

/** Stable error codes from the server (see docs/agent-protocol.md). */
export type ErrorCode =
  | "invalid_request"
  | "unauthorized"
  | "remote_disabled"
  | "auth_unavailable"
  | "not_found"
  | "nothing_published"
  | "job_queue_full"
  | "rate_limited"
  | "build_already_running"
  | "job_not_cancellable"
  | "replan_required"
  | "cancelled"
  | "llm_error"
  | "storage_error"
  | "internal_error"
  | (string & {});

export class LlmWikiApiError extends Error {
  readonly status: number;
  readonly code: ErrorCode;
  /** Echoed `x-request-id` — attach to bug reports. */
  readonly requestId?: string;

  constructor(status: number, code: ErrorCode, message: string, requestId?: string) {
    super(`llm-wiki ${code} (${status}): ${message}`);
    this.name = "LlmWikiApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

export interface KnowledgeProviderOptions {
  /** Base URL of the llm-wiki server, e.g. "http://127.0.0.1:8080". */
  baseUrl: string;
  /** Bearer token for remote-mode servers. */
  apiKey?: string;
  /** Request timeout in milliseconds (default 120_000). */
  timeoutMs?: number;
  /** Fetch implementation override (tests, custom agents). */
  fetch?: typeof fetch;
}

// ---------------------------------------------------------------------------
// Response types (frozen at protocol v1)
// ---------------------------------------------------------------------------

export interface ServerStatus {
  protocol_version: number;
  workspace: string;
  sources: number;
  latest_build: { build_id: string; status: string; started_at: string } | null;
  active_build_id: string | null;
  jobs: Record<string, number>;
}

export interface JobRecord {
  job_id: string;
  kind: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED" | "INTERRUPTED" | "REPLAN_REQUIRED";
  phase: string | null;
  build_id: string | null;
  failure_code: string | null;
  retryable: boolean;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface SearchHit {
  page_id: string;
  slug: string;
  title: string;
  heading_path: string[];
  snippet: string;
  rank: number;
  citation_count: number;
}

export interface SearchResponse {
  protocol_version: number;
  request_id: string;
  generation: string | null;
  hits: SearchHit[];
  truncated: boolean;
}

export interface ContextChunk {
  slug: string;
  title: string;
  heading_path: string[];
  snippet: string;
  score: number;
  sources: string[];
}

export interface ContextNeighbor {
  from_slug: string;
  node_id: string;
  node_type: string;
  label: string;
  relation: string;
}

export interface ContextBudget {
  max_chunks?: number;
  max_tokens?: number;
  max_pages?: number;
  max_per_source?: number;
  graph_limit?: number;
}

export interface ContextResponse {
  protocol_version: number;
  request_id: string;
  generation: string | null;
  chunks: ContextChunk[];
  neighbors: ContextNeighbor[];
  estimated_tokens: number;
  dropped: number;
  truncated: boolean;
}

export interface QueryCitation {
  claim_node_id: string;
  source_id: string;
  section_id: string | null;
  source_hash: string;
  evidence_digest: string;
  heading_path: string[];
}

export interface QueryResponse {
  protocol_version: number;
  request_id: string;
  generation: string | null;
  answer: string;
  citations: QueryCitation[];
  sources: string[];
  llm_request_count: number;
  insight_id: string | null;
}

export interface InsightCitation {
  claim_node_id: string;
  source: string;
  heading_path: string[];
  range: [number, number];
  evidence_digest: string;
}

export interface Insight {
  insight_id: string;
  build_id: string;
  query: string;
  answer: string;
  citations: InsightCitation[];
  created_at: string;
}

export interface InsightsResponse {
  protocol_version: number;
  request_id: string;
  insights: Insight[];
  next_cursor: string | null;
  truncated: boolean;
}

export interface PageSummary {
  page_id: string;
  slug: string;
  title: string;
  category: string;
  language: string;
  body_hash: string;
}

export interface PagesResponse {
  generation: string;
  pages: PageSummary[];
  next_cursor: string | null;
  truncated: boolean;
}

export interface BuildAccepted {
  job_id: string;
  status: "queued";
}

// ---------------------------------------------------------------------------
// The KnowledgeProvider
// ---------------------------------------------------------------------------

export interface KnowledgeProvider {
  status(): Promise<ServerStatus>;
  search(query: string, limit?: number): Promise<SearchResponse>;
  context(query: string, budget?: ContextBudget, hybrid?: boolean): Promise<ContextResponse>;
  ask(query: string, opts?: { writeBack?: boolean; hybrid?: boolean; embeddingModel?: string }): Promise<QueryResponse>;
  insights(limit?: number, cursor?: string): Promise<InsightsResponse>;
  pages(limit?: number, cursor?: string): Promise<PagesResponse>;
  page(idOrSlug: string): Promise<Record<string, unknown>>;
  build(sourceId?: string, idempotencyKey?: string): Promise<BuildAccepted>;
  job(jobId: string): Promise<JobRecord>;
  cancelJob(jobId: string): Promise<{ job_id: string; status: string }>;
}

/** The single protocol major version this SDK understands. */
const SUPPORTED_PROTOCOL_VERSION = 1;

/**
 * Rejects responses whose `protocol_version` has a different MAJOR version
 * instead of guessing at an evolved contract (README guarantee). Responses
 * without the field (e.g. `/health`) pass through untouched.
 */
function assertProtocolVersion(payload: unknown): void {
  if (
    payload !== null &&
    typeof payload === "object" &&
    "protocol_version" in payload
  ) {
    const version = (payload as { protocol_version?: unknown }).protocol_version;
    if (typeof version !== "number" || version !== SUPPORTED_PROTOCOL_VERSION) {
      throw new LlmWikiApiError(
        0,
        "protocol_mismatch",
        `server speaks protocol ${String(version)}; this SDK understands ${SUPPORTED_PROTOCOL_VERSION}. Upgrade @llm-wiki/sdk.`,
      );
    }
  }
}

export class LlmWikiClient implements KnowledgeProvider {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: KnowledgeProviderOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 120_000;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  async status(): Promise<ServerStatus> {
    return this.get<ServerStatus>("/v1/status");
  }

  async search(query: string, limit = 10): Promise<SearchResponse> {
    return this.post<SearchResponse>("/v1/search", { query, limit });
  }

  async context(query: string, budget?: ContextBudget, hybrid = false): Promise<ContextResponse> {
    return this.post<ContextResponse>("/v1/context", { query, budget, hybrid });
  }

  async ask(
    query: string,
    opts?: { writeBack?: boolean; hybrid?: boolean; embeddingModel?: string },
  ): Promise<QueryResponse> {
    return this.post<QueryResponse>("/v1/query", {
      query,
      write_back: opts?.writeBack ?? false,
      hybrid: opts?.hybrid ?? false,
      embedding_model: opts?.embeddingModel,
    });
  }

  async insights(limit = 20, cursor?: string): Promise<InsightsResponse> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (cursor) params.set("cursor", cursor);
    return this.get<InsightsResponse>(`/v1/insights?${params.toString()}`);
  }

  async pages(limit = 20, cursor?: string): Promise<PagesResponse> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (cursor) params.set("cursor", cursor);
    return this.get<PagesResponse>(`/v1/pages?${params.toString()}`);
  }

  async page(idOrSlug: string): Promise<Record<string, unknown>> {
    return this.get<Record<string, unknown>>(`/v1/pages/${encodeURIComponent(idOrSlug)}`);
  }

  async build(sourceId = "default", idempotencyKey?: string): Promise<BuildAccepted> {
    return this.post<BuildAccepted>(
      "/v1/build",
      { source_id: sourceId },
      idempotencyKey ? { "idempotency-key": idempotencyKey } : undefined,
    );
  }

  async job(jobId: string): Promise<JobRecord> {
    return this.get<JobRecord>(`/v1/jobs/${encodeURIComponent(jobId)}`);
  }

  async cancelJob(jobId: string): Promise<{ job_id: string; status: string }> {
    return this.post<{ job_id: string; status: string }>(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {});
  }

  // ---- transport ----

  private async request<T>(method: string, path: string, body?: unknown, headers?: Record<string, string>): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers: {
          ...(body !== undefined ? { "content-type": "application/json" } : {}),
          ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}),
          ...headers,
        },
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      const requestId = response.headers.get("x-request-id") ?? undefined;
      const text = await response.text();
      if (!response.ok) {
        let code: ErrorCode = "internal_error";
        let message = text;
        try {
          const parsed = JSON.parse(text) as { error?: { code?: string; message?: string } };
          code = parsed.error?.code ?? code;
          message = parsed.error?.message ?? message;
        } catch {
          // keep raw text as the message
        }
        throw new LlmWikiApiError(response.status, code, message, requestId);
      }
      const parsed = JSON.parse(text) as T;
      assertProtocolVersion(parsed);
      return parsed;
    } finally {
      clearTimeout(timer);
    }
  }

  private get<T>(path: string): Promise<T> {
    return this.request<T>("GET", path);
  }

  private post<T>(path: string, body: unknown, headers?: Record<string, string>): Promise<T> {
    return this.request<T>("POST", path, body, headers);
  }
}
