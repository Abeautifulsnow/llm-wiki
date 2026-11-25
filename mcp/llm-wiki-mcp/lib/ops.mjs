/**
 * Operations: one function per tool/resource, transport-agnostic.
 *
 * Everything returns a model-facing Markdown string (or a resource payload),
 * so the stdio transports stay thin protocol adapters and both of them emit
 * byte-identical tool results.
 */

import { randomUUID } from "node:crypto";
import { PAGE_URI_PREFIX } from "./constants.mjs";
import { config, log } from "./config.mjs";
import { createApi, clampInt, normalizeMode } from "./api.mjs";
import {
  renderAnswer,
  renderContext,
  renderJob,
  renderPage,
  renderSearch,
  renderStatus,
} from "./render.mjs";

/** Hard ceiling for the `resources/list` walk: 20 × 100 pages = 2000 pages. */
const PAGE_LIST_MAX_ROUNDS = 20;

function pageToResource(page) {
  return {
    uri: pageUri(page.slug),
    name: page.title ?? page.slug,
    title: page.title ?? page.slug,
    mimeType: "text/markdown",
    description: [page.category, page.language].filter(Boolean).join(" · ") || undefined,
  };
}

/** Raised when a `wiki://page/<slug>` URI is malformed or points nowhere. */
export class ResourceNotFoundError extends Error {
  constructor(message, uri) {
    super(message);
    this.uri = uri;
  }
}

// Re-exported for the SDK build, which builds its ResourceTemplate from it.
export { PAGE_URI_PREFIX };

export function pageUri(slug) {
  return `${PAGE_URI_PREFIX}${encodeURIComponent(slug ?? "")}`;
}

export function slugFromUri(uri) {
  if (typeof uri !== "string" || !uri.startsWith(PAGE_URI_PREFIX)) return null;
  const raw = uri.slice(PAGE_URI_PREFIX.length);
  if (raw === "") return null;
  try {
    return decodeURIComponent(raw);
  } catch {
    return null;
  }
}

export function createOps(api = createApi(config)) {
  const { callApi, withSourceModeFallback } = api;

  return {
    async status() {
      return renderStatus(await callApi("/v1/status"));
    },

    async context(args) {
      const mode = normalizeMode(args.source_mode, config.sourceMode);
      const body = {
        query: args.query,
        hybrid: args.hybrid === true,
        source_mode: mode,
        budget: {
          max_chunks: clampInt(args.max_chunks, { min: 1, max: 50, fallback: config.maxChunks }),
          max_tokens: clampInt(args.max_tokens, { min: 500, max: 50000, fallback: config.maxTokens }),
          max_pages: clampInt(args.max_pages, { min: 1, max: 20, fallback: 4 }),
          max_per_source: clampInt(args.max_per_source, { min: 1, max: 20, fallback: 2 }),
          graph_limit: clampInt(args.graph_limit, { min: 0, max: 50, fallback: 4 }),
        },
      };
      const data = await withSourceModeFallback(mode, (effectiveMode) =>
        callApi("/v1/context", { method: "POST", body: { ...body, source_mode: effectiveMode } }),
      );
      return renderContext(data);
    },

    async search(args) {
      const mode = normalizeMode(args.source_mode, config.sourceMode);
      const limit = clampInt(args.limit, { min: 1, max: 50, fallback: 10 });
      const data = await withSourceModeFallback(mode, (effectiveMode) =>
        callApi("/v1/search", { method: "POST", body: { query: args.query, limit, source_mode: effectiveMode } }),
      );
      return renderSearch(data);
    },

    async page(id) {
      return renderPage(await callApi(`/v1/pages/${encodeURIComponent(id)}`));
    },

    async ask(args) {
      const body = {
        query: args.query,
        write_back: args.write_back === true,
        hybrid: args.hybrid === true,
      };
      return renderAnswer(await callApi("/v1/query", { method: "POST", body }));
    },

    async build(args = {}) {
      const key = args.idempotency_key ?? `llm-wiki-mcp:${randomUUID()}`;
      const data = await callApi("/v1/build", {
        method: "POST",
        body: { source_id: "default" },
        headers: { "idempotency-key": key },
      });
      const replayed = data.replayed ? " · replayed" : "";
      return `job: ${data.job_id ?? "?"} · status: ${data.status ?? "?"}${replayed}\nPoll with \`wiki_job\`.`;
    },

    async job(jobId) {
      return renderJob(await callApi(`/v1/jobs/${encodeURIComponent(jobId)}`));
    },

    /**
     * Page list backing `resources/list`; an unpublished workspace is empty,
     * not an error. Walks `next_cursor` to the end: a fixed `limit=100` would
     * silently hide every page past the hundredth, and MCP gives us no way to
     * tell the client that the list is partial.
     */
    async listPageResources() {
      const collected = [];
      let cursor = null;
      for (let round = 0; round < PAGE_LIST_MAX_ROUNDS; round += 1) {
        const query = cursor ? `?limit=100&cursor=${encodeURIComponent(cursor)}` : "?limit=100";
        let data;
        try {
          data = await callApi(`/v1/pages${query}`);
        } catch (error) {
          if (error?.code === "nothing_published") return [];
          throw error;
        }
        collected.push(...(Array.isArray(data.pages) ? data.pages : []));
        if (!data.truncated || !data.next_cursor) return collected.map(pageToResource);
        cursor = data.next_cursor;
      }
      log(`resources/list stopped after ${PAGE_LIST_MAX_ROUNDS} rounds (${collected.length} pages); the wiki is larger than expected`);
      return collected.map(pageToResource);
    },

    /** Page body backing `resources/read`. */
    async readPageResource(uri) {
      const slug = slugFromUri(uri);
      if (!slug) throw new ResourceNotFoundError(`unsupported resource uri: ${String(uri)}`, uri);
      let data;
      try {
        data = await callApi(`/v1/pages/${encodeURIComponent(slug)}`);
      } catch (error) {
        if (error?.code === "not_found") {
          throw new ResourceNotFoundError(`no page '${slug}' in the active generation`, uri);
        }
        throw error;
      }
      return { uri, mimeType: "text/markdown", text: String(data.content ?? "") };
    },
  };
}
