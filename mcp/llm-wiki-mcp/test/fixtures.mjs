/**
 * Shared test rig: a fake llm-wiki HTTP API plus an official-SDK MCP client
 * wired to one build over stdio.
 *
 * Kept separate from the existing suites so new behaviour tests can reuse it
 * without touching the hand-rolled client in `handshake.test.mjs`.
 */

import { createServer } from "node:http";
import { once } from "node:events";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const CONTEXT = {
  protocol_version: 1,
  request_id: "req_behaviour",
  generation: "bld_behaviour",
  chunks: [
    {
      slug: "auth/sso-flow",
      title: "SSO flow",
      heading_path: ["Auth", "SSO"],
      snippet: "The client redirects to the IdP, then exchanges the code for a token.",
      score: 0.87,
      sources: ["docs/auth.md"],
      evidence_kind: "wiki",
      source_ref: null,
    },
  ],
  neighbors: [],
  estimated_tokens: 210,
  dropped: 0,
  truncated: false,
  served_mode: "fusion",
  degraded: { wiki: "served", source: "served" },
};

const PAGE = {
  page_id: "wp_auth",
  slug: "auth/sso-flow",
  title: "SSO flow",
  category: "auth",
  language: "en",
  body_hash: "h",
  content: "# SSO flow\n\nThe client redirects to the IdP.",
  knowledge_refs: [],
  citations: [
    {
      claim_node_id: "cl_1",
      source_id: "src_auth",
      section_id: "sec_1",
      range: [10, 80],
      source_hash: "sh",
      evidence_digest: "0123456789abcdef",
      heading_path: ["Auth"],
    },
  ],
  links: [],
};

/**
 * Starts the fake API. `state` records what the server actually received, so
 * tests can assert on wire-level details (encoded paths, headers, queries).
 */
export function startFakeWiki() {
  const state = { pagePaths: [], contextQueries: [], buildKeys: [] };

  const server = createServer(async (request, response) => {
    response.on("error", () => {});
    const url = new URL(request.url, "http://127.0.0.1");
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const raw = Buffer.concat(chunks).toString("utf8");
    let body = null;
    try {
      body = raw === "" ? null : JSON.parse(raw);
    } catch {
      body = null;
    }

    const json = (status, payload) => {
      if (response.writableEnded || response.destroyed) return;
      response.writeHead(status, { "content-type": "application/json", "x-request-id": "req_behaviour" });
      response.end(JSON.stringify(payload));
    };

    if (request.method === "GET" && url.pathname === "/v1/status") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_behaviour",
        workspace: "behaviour-ws",
        sources: 2,
        latest_build: { build_id: "bld_behaviour", status: "COMPLETED", started_at: "2026-01-01T00:00:00Z" },
        active_build_id: "bld_behaviour",
        jobs: {},
        versions: { binary: "0.3.0", generation_compiler: "0.3.0", db_schema: 14, generation_stale: false },
      });
    }

    if (request.method === "POST" && url.pathname === "/v1/context") {
      state.contextQueries.push(body?.query);
      // Longer than any timeout the server accepts (its floor is 1000ms), so the
      // timeout test never races the fake response.
      if (body?.query === "slow") await sleep(2500);
      if (body?.query === "degraded") {
        return json(200, { ...CONTEXT, degraded: { wiki: "not_published", source: "served" } });
      }
      return json(200, CONTEXT);
    }

    if (request.method === "GET" && url.pathname === "/v1/pages") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_behaviour",
        generation: "bld_behaviour",
        pages: [{ page_id: "wp_auth", slug: "auth/sso-flow", title: "SSO flow", category: "auth", language: "en", body_hash: "h" }],
        next_cursor: null,
        truncated: false,
      });
    }

    if (request.method === "GET" && url.pathname.startsWith("/v1/pages/")) {
      state.pagePaths.push(request.url);
      const slug = decodeURIComponent(url.pathname.slice("/v1/pages/".length));
      if (slug === "auth/sso-flow") return json(200, PAGE);
      return json(404, { error: { code: "not_found", message: "no page" } });
    }

    if (request.method === "POST" && url.pathname === "/v1/query") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_behaviour",
        generation: "bld_behaviour",
        answer: "Grounded answer.",
        citations: [{ claim_node_id: "cl_1", source: "docs/auth.md", heading_path: ["Auth"], range: [10, 80], evidence_digest: "0123456789abcdef" }],
        sources: ["docs/auth.md"],
        llm_request_count: 1,
        insight_id: "ins_1",
      });
    }

    if (request.method === "POST" && url.pathname === "/v1/build") {
      state.buildKeys.push(request.headers["idempotency-key"]);
      return json(202, { job_id: "job_1", status: "queued", replayed: true });
    }

    if (request.method === "GET" && url.pathname === "/v1/jobs/job_1") {
      return json(200, {
        job_id: "job_1",
        kind: "build",
        status: "RUNNING",
        phase: "COMPILING",
        failure_code: null,
        retryable: false,
        error: null,
        progress: { pages: 3 },
        created_at: "2026-01-01T00:00:00Z",
        started_at: "2026-01-01T00:00:01Z",
        finished_at: null,
      });
    }

    return json(404, { error: { code: "not_found", message: `unhandled ${request.method} ${url.pathname}` } });
  });

  return { server, state };
}

/** Connects the official SDK client to `buildFile` and runs `run`. */
export async function withClient(buildFile, run, extraEnv = {}) {
  const fake = startFakeWiki();
  fake.server.listen(0, "127.0.0.1");
  await once(fake.server, "listening");
  const baseUrl = `http://127.0.0.1:${fake.server.address().port}`;

  const env = Object.fromEntries(
    Object.entries({ ...process.env, LLM_WIKI_BASE_URL: baseUrl, ...extraEnv }).filter(
      ([, value]) => value !== undefined,
    ),
  );

  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [buildFile],
    env,
    stderr: "pipe",
  });
  const client = new Client({ name: "llm-wiki-behaviour", version: "0.0.0" }, { capabilities: {} });
  await client.connect(transport);

  try {
    await run({ client, state: fake.state, baseUrl });
  } finally {
    await client.close().catch(() => {});
    fake.server.closeAllConnections?.();
    fake.server.close();
    await once(fake.server, "close").catch(() => {});
  }
}

/** Joins every text block of a tool result. */
export function textOf(result) {
  return (result?.content ?? [])
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}
