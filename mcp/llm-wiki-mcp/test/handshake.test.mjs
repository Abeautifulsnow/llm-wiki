/**
 * End-to-end test for the llm-wiki MCP server.
 *
 * Boots a fake llm-wiki HTTP API (protocol v1) on an ephemeral port, spawns the
 * real stdio server against it, and drives a full MCP session: initialize →
 * tools/list → tools/call → resources/list → resources/read, plus the error
 * paths (invalid params, unknown tool, upstream 500, legacy server without
 * `source_mode`).
 *
 * Run: node --test test
 */

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import test from "node:test";

const HERE = dirname(fileURLToPath(import.meta.url));
const SERVER = join(HERE, "..", "server.mjs");

// ---------------------------------------------------------------------------
// Fake llm-wiki API
// ---------------------------------------------------------------------------

const CONTEXT_PAYLOAD = {
  protocol_version: 1,
  request_id: "req_test",
  generation: "bld_test",
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
    {
      slug: "docs/auth.md",
      title: "auth.md",
      heading_path: ["Token exchange"],
      snippet: "POST /token with code and verifier.",
      score: 0.61,
      sources: ["docs/auth.md"],
      evidence_kind: "source",
      source_ref: {
        source_id: "src_auth",
        file_path: "docs/auth.md",
        heading_path: ["Token exchange"],
        ordinal: 2,
        range_start: 120,
        range_end: 260,
      },
    },
  ],
  neighbors: [{ from_slug: "auth/sso-flow", node_id: "n1", node_type: "concept", label: "OAuth", relation: "uses" }],
  estimated_tokens: 321,
  dropped: 0,
  truncated: false,
  served_mode: "fusion",
  degraded: { wiki: "served", source: "served" },
};

const PAGE_PAYLOAD = {
  page_id: "wp_auth",
  slug: "auth/sso-flow",
  title: "SSO flow",
  category: "auth",
  language: "en",
  body_hash: "hash",
  content: "# SSO flow\n\nThe client redirects to the IdP.",
  knowledge_refs: ["kn_1"],
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
  links: [{ to_page_id: "wp_token", target_title: "Token exchange" }],
};

function startFakeWiki({ rejectSourceMode = false } = {}) {
  const calls = [];
  const server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const body = await readBody(request);
    calls.push({ method: request.method, path: url.pathname, body });

    const json = (status, payload) => {
      response.writeHead(status, { "content-type": "application/json", "x-request-id": "req_test" });
      response.end(JSON.stringify(payload));
    };

    if (rejectSourceMode && body?.source_mode && body.source_mode !== "wiki") {
      return json(400, {
        error: { code: "invalid_request", message: "unknown field `source_mode`" },
      });
    }

    if (request.method === "GET" && url.pathname === "/v1/status") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_test",
        workspace: "test-ws",
        sources: 3,
        latest_build: { build_id: "bld_test", status: "COMPLETED", started_at: "2026-01-01T00:00:00Z" },
        active_build_id: "bld_test",
        jobs: { QUEUED: 0, RUNNING: 0 },
        versions: { binary: "0.1.0", generation_compiler: "0.1.0", db_schema: 14, generation_stale: false },
      });
    }
    if (request.method === "POST" && url.pathname === "/v1/context") {
      if (body?.query === "fail") {
        return json(500, { error: { code: "storage_error", message: "disk on fire" } });
      }
      return json(200, CONTEXT_PAYLOAD);
    }
    if (request.method === "POST" && url.pathname === "/v1/search") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_test",
        generation: "bld_test",
        source_mode: "wiki",
        served: { wiki: "served", source: "disabled" },
        hits: [
          {
            page_id: "wp_auth",
            slug: "auth/sso-flow",
            title: "SSO flow",
            heading_path: ["Auth"],
            snippet: "redirect to the IdP",
            rank: -1.2,
            citation_count: 2,
            evidence_kind: "wiki",
          },
        ],
        truncated: false,
      });
    }
    if (request.method === "GET" && url.pathname === "/v1/pages") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_test",
        generation: "bld_test",
        pages: [{ page_id: "wp_auth", slug: "auth/sso-flow", title: "SSO flow", category: "auth", language: "en", body_hash: "h" }],
        next_cursor: null,
        truncated: false,
      });
    }
    if (request.method === "GET" && url.pathname.startsWith("/v1/pages/")) {
      const slug = decodeURIComponent(url.pathname.slice("/v1/pages/".length));
      if (slug === "auth/sso-flow") return json(200, PAGE_PAYLOAD);
      return json(404, { error: { code: "not_found", message: "no page" } });
    }
    if (request.method === "POST" && url.pathname === "/v1/build") {
      return json(202, { job_id: "job_1", status: "queued" });
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

  return { server, calls };
}

async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8");
  if (raw === "") return null;
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

// ---------------------------------------------------------------------------
// Minimal MCP client over stdio
// ---------------------------------------------------------------------------

async function startClient(baseUrl, extraEnv = {}) {
  const child = spawn(process.execPath, [SERVER], {
    env: { ...process.env, LLM_WIKI_BASE_URL: baseUrl, ...extraEnv },
    stdio: ["pipe", "pipe", "pipe"],
  });

  const pending = new Map();
  const notifications = [];
  let nextId = 1;
  let stderr = "";

  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  lines.on("line", (line) => {
    const message = JSON.parse(line);
    if (message.id === undefined || message.id === null) {
      notifications.push(message);
      return;
    }
    const waiter = pending.get(message.id);
    if (!waiter) return;
    pending.delete(message.id);
    waiter(message);
  });

  await once(child, "spawn");

  function send(message) {
    child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  function request(method, params, timeoutMs = 5000) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`timeout waiting for ${method}; stderr=${stderr}`));
      }, timeoutMs);
      pending.set(id, (message) => {
        clearTimeout(timer);
        resolve(message);
      });
      send({ jsonrpc: "2.0", id, method, params });
    });
  }

  function notify(method, params) {
    send({ jsonrpc: "2.0", method, params });
  }

  async function close() {
    child.stdin.end();
    await Promise.race([once(child, "exit"), new Promise((resolve) => setTimeout(resolve, 1000))]);
    if (child.exitCode === null) child.kill();
    lines.close();
  }

  return { child, request, notify, close, notifications, stderr: () => stderr };
}

async function withRig(options, run) {
  const fake = startFakeWiki(options);
  fake.server.listen(0, "127.0.0.1");
  await once(fake.server, "listening");
  const { port } = fake.server.address();
  const client = await startClient(`http://127.0.0.1:${port}`, options?.env);
  try {
    await run({ client, calls: fake.calls, baseUrl: `http://127.0.0.1:${port}` });
  } finally {
    await client.close();
    fake.server.close();
    await once(fake.server, "close").catch(() => {});
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test("initialize negotiates the protocol version and advertises tools", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("initialize", {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "test", version: "0" },
    });
    assert.equal(response.result.protocolVersion, "2025-06-18");
    assert.equal(response.result.serverInfo.name, "llm-wiki-mcp");
    assert.deepEqual(response.result.capabilities.tools, { listChanged: false });
    assert.match(response.result.instructions, /retrieval layer/);
    client.notify("notifications/initialized");
  });
});

test("initialize falls back to the latest known version for an unknown request version", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("initialize", {
      protocolVersion: "1999-01-01",
      capabilities: {},
      clientInfo: { name: "test", version: "0" },
    });
    assert.equal(response.result.protocolVersion, "2025-06-18");
  });
});

test("tools/list exposes the four read-only tools by default", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("tools/list");
    const names = response.result.tools.map((tool) => tool.name).sort();
    assert.deepEqual(names, ["wiki_context", "wiki_page", "wiki_search", "wiki_status"]);
    for (const tool of response.result.tools) {
      assert.equal(tool.inputSchema.type, "object");
      assert.ok(tool.description.length > 40);
    }
  });
});

test("wiki_ask and wiki_build are opt-in", async () => {
  await withRig({ env: { LLM_WIKI_MCP_ENABLE_ASK: "1", LLM_WIKI_MCP_ENABLE_BUILD: "1" } }, async ({ client }) => {
    const response = await client.request("tools/list");
    const names = response.result.tools.map((tool) => tool.name);
    assert.ok(names.includes("wiki_ask"));
    assert.ok(names.includes("wiki_build"));
    assert.ok(names.includes("wiki_job"));
  });
});

test("wiki_context renders chunks with provenance and sends the default budget", async () => {
  await withRig({}, async ({ client, calls }) => {
    const response = await client.request("tools/call", {
      name: "wiki_context",
      arguments: { query: "how does sso work" },
    });
    const text = response.result.content[0].text;
    assert.equal(response.result.isError, false);
    assert.match(text, /### 1\. SSO flow/);
    assert.match(text, /location: auth\/sso-flow › Auth › SSO/);
    assert.match(text, /docs\/auth\.md:120-260/);
    assert.match(text, /Related \(wiki graph\)/);
    assert.match(text, /Cite the `location`/);

    const sent = calls.find((call) => call.path === "/v1/context");
    assert.equal(sent.body.source_mode, "fusion");
    assert.equal(sent.body.budget.max_chunks, 8);
    assert.equal(sent.body.budget.max_tokens, 8000);
  });
});

test("wiki_context honours explicit arguments", async () => {
  await withRig({}, async ({ client, calls }) => {
    await client.request("tools/call", {
      name: "wiki_context",
      arguments: { query: "sso", source_mode: "wiki", max_chunks: 3, max_tokens: 2000, hybrid: true },
    });
    const sent = calls.find((call) => call.path === "/v1/context");
    assert.equal(sent.body.source_mode, "wiki");
    assert.equal(sent.body.budget.max_chunks, 3);
    assert.equal(sent.body.hybrid, true);
  });
});

test("a server without source_mode support is retried in wiki mode", async () => {
  await withRig({ rejectSourceMode: true }, async ({ client, calls }) => {
    const response = await client.request("tools/call", {
      name: "wiki_context",
      arguments: { query: "sso" },
    });
    assert.equal(response.result.isError, false);
    const modes = calls.filter((call) => call.path === "/v1/context").map((call) => call.body.source_mode);
    assert.deepEqual(modes, ["fusion", "wiki"]);
  });
});

test("wiki_search renders ranked hits", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("tools/call", {
      name: "wiki_search",
      arguments: { query: "sso", limit: 5 },
    });
    const text = response.result.content[0].text;
    assert.match(text, /1\. SSO flow/);
    assert.match(text, /kind: wiki/);
    assert.match(text, /These are page\/section hits/);
  });
});

test("wiki_page returns the body, citations and links", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("tools/call", {
      name: "wiki_page",
      arguments: { id: "auth/sso-flow" },
    });
    const text = response.result.content[0].text;
    assert.match(text, /# SSO flow/);
    assert.match(text, /Citations \(1\)/);
    assert.match(text, /src_auth:10-80/);
    assert.match(text, /Token exchange/);
    assert.match(text, /0123456789ab/);
  });
});

test("wiki_status explains an unpublished workspace", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("tools/call", { name: "wiki_status", arguments: {} });
    const text = response.result.content[0].text;
    assert.match(text, /workspace: test-ws/);
    assert.match(text, /active generation: bld_test/);
  });
});

test("upstream 500 surfaces as isError with the stable code", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("tools/call", {
      name: "wiki_context",
      arguments: { query: "fail" },
    });
    assert.equal(response.result.isError, true);
    assert.match(response.result.content[0].text, /storage_error/);
    assert.match(response.result.content[0].text, /HTTP 500/);
  });
});

test("missing query and unknown tools are protocol errors", async () => {
  await withRig({}, async ({ client }) => {
    const missing = await client.request("tools/call", { name: "wiki_context", arguments: {} });
    assert.equal(missing.error.code, -32602);
    assert.match(missing.error.message, /query/);

    const unknown = await client.request("tools/call", { name: "nope", arguments: {} });
    assert.equal(unknown.error.code, -32602);
    assert.match(unknown.error.message, /unknown tool/);
  });
});

test("an invalid source_mode is rejected before reaching the API", async () => {
  await withRig({}, async ({ client, calls }) => {
    const response = await client.request("tools/call", {
      name: "wiki_context",
      arguments: { query: "sso", source_mode: "vector" },
    });
    assert.equal(response.error.code, -32602);
    assert.equal(calls.filter((call) => call.path === "/v1/context").length, 0);
  });
});

test("resources/list and resources/read expose published pages", async () => {
  await withRig({}, async ({ client }) => {
    const list = await client.request("resources/list");
    assert.equal(list.result.resources.length, 1);
    const resource = list.result.resources[0];
    assert.equal(resource.uri, "wiki://page/auth%2Fsso-flow");
    assert.equal(resource.mimeType, "text/markdown");

    const read = await client.request("resources/read", { uri: resource.uri });
    assert.match(read.result.contents[0].text, /# SSO flow/);
  });
});

test("resources/read reports an unknown page as resource-not-found", async () => {
  await withRig({}, async ({ client }) => {
    const response = await client.request("resources/read", { uri: "wiki://page/does-not-exist" });
    assert.equal(response.error.code, -32002);
  });
});

test("an unreachable llm-wiki server yields an actionable message", async () => {
  const client = await startClient("http://127.0.0.1:9", { LLM_WIKI_MCP_TIMEOUT_MS: "1500" });
  try {
    const response = await client.request("tools/call", { name: "wiki_status", arguments: {} });
    assert.equal(response.result.isError, true);
    assert.match(response.result.content[0].text, /cannot reach|unreachable/);
    assert.match(response.result.content[0].text, /llm-wiki serve/);
  } finally {
    await client.close();
  }
});

test("malformed JSON is reported as a parse error without killing the session", async () => {
  await withRig({}, async ({ client }) => {
    client.child.stdin.write("{not json}\n");
    // A parse error has no request id, so it arrives with `id: null` — the
    // client records id-less frames as notifications.
    const deadline = Date.now() + 2000;
    while (Date.now() < deadline && !client.notifications.some((m) => m.error?.code === -32700)) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    const parseError = client.notifications.find((m) => m.error?.code === -32700);
    assert.ok(parseError, `expected a -32700 frame, got ${JSON.stringify(client.notifications)}`);
    assert.match(parseError.error.message, /invalid JSON/);

    // The session survives a malformed frame.
    const response = await client.request("tools/list");
    assert.equal(response.result.tools.length, 4);
  });
});
