/**
 * Embedding-job surface for the KnowledgeProvider SDK (LW-003).
 *
 * `POST /v1/embed` and its async counterpart `POST /v1/embed/jobs` are an
 * ADDITIVE protocol surface (0014): the frozen protocol-v1 entry
 * (`./index.ts`) is unchanged, so the methods live here and are attached by
 * [`withEmbeddingJobs`] — no consumer of the v1 client is affected, and
 * folding them into `LlmWikiClient` later stays source-compatible because the
 * extended client IS a `LlmWikiClient`.
 *
 * Usage:
 *
 * ```ts
 * import { LlmWikiClientWithEmbeddingJobs } from "@llm-wiki/sdk/embed";
 *
 * const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: "http://127.0.0.1:8080" });
 * const ack = await wiki.startEmbedJob("bge-m3", { idempotencyKey: "nightly-1" });
 * const job = await wiki.job(ack.job_id);      // job.progress — the embed counters
 * await wiki.cancelJob(ack.job_id);            // cooperative stop
 * ```
 */

import { LlmWikiClient } from "./index.js";

/** Live counters of an EMBED job (additive, 0014). `covered_before` is the
 * section count that already had a vector for this model, so `embedded` is
 * what the run actually spent model calls on. */
export interface EmbedProgress {
  total: number;
  covered_before: number;
  embedded: number;
}

/** `POST /v1/embed` (frozen protocol v1): the SYNCHRONOUS backfill report —
 * the request stays open for the whole run. Prefer `startEmbedJob` for large
 * corpora, where a client timeout loses the outcome. */
export interface EmbedReport {
  protocol_version: number;
  request_id: string;
  generation: string | null;
  model: string;
  total_sections: number;
  covered_before: number;
  embedded: number;
}

/** `POST /v1/embed/jobs` (additive, 0014): the async submission ack. Poll
 * `job(job_id)` — its `progress` carries the embed counters — and stop it
 * cooperatively with `cancelJob(job_id)`. */
export interface EmbedJobAccepted {
  job_id: string;
  status: "queued";
}

/** Optional overrides for an embedding backfill. */
export interface EmbedOptions {
  /** Sections per embedding request (server default: `[embedding] batch_size`). */
  batch?: number;
  /** Replay-safe submission key, sent as `Idempotency-Key` and scoped by the
   * server to `embed:{generation}:{model}:{key}`: retrying with the same key
   * resolves to the SAME job, while a new generation or model starts a fresh
   * one. Omit it and a retry may spend model calls twice. */
  idempotencyKey?: string;
}

/**
 * The slice of the client transport these methods need, declared structurally
 * because `LlmWikiClient` keeps its transport private. Going through the same
 * `post` keeps timeouts, bearer auth, `protocol_version` assertion and
 * `LlmWikiApiError` mapping identical to every other SDK call — the single
 * cast in the mixin is the price of not widening the v1 client's surface.
 */
interface PostCapable {
  post<T>(path: string, body: unknown, headers?: Record<string, string>): Promise<T>;
}

/** Attaches the embedding-job surface to any `LlmWikiClient` subclass. */
export function withEmbeddingJobs<TBase extends new (...args: any[]) => LlmWikiClient>(
  Base: TBase,
) {
  return class EmbeddingJobs extends Base {
    /** Synchronous backfill: resolves when the run finishes (frozen v1). */
    async embed(model?: string, options?: EmbedOptions): Promise<EmbedReport> {
      return (this as unknown as PostCapable).post<EmbedReport>("/v1/embed", {
        model,
        batch: options?.batch,
      });
    }

    /** Async backfill: returns as soon as the job is admitted (202), then
     * `job(job_id)` for progress and `cancelJob(job_id)` to stop it. */
    async startEmbedJob(model?: string, options?: EmbedOptions): Promise<EmbedJobAccepted> {
      return (this as unknown as PostCapable).post<EmbedJobAccepted>(
        "/v1/embed/jobs",
        { model, batch: options?.batch },
        options?.idempotencyKey ? { "idempotency-key": options.idempotencyKey } : undefined,
      );
    }
  };
}

/** The protocol-v1 client with the embedding-job surface attached. */
export class LlmWikiClientWithEmbeddingJobs extends withEmbeddingJobs(LlmWikiClient) {}
