/**
 * Contract tests for the embedding-job SDK surface (LW-003).
 *
 * These pin the WIRE, not the implementation: exact method, path, headers and
 * JSON body for every embed-related call, plus the error/protocol contracts the
 * README promises. The server side of each expectation lives in
 * `crates/llm-wiki-server/src/routes.rs` (`embed`, `embed_job`) and is
 * documented in `docs/agent-protocol.md` — if the two drift, one of these
 * tests must fail.
 *
 * Plain JS on purpose: the SDK ships source-only (consumers compile it), so the
 * tests compile with `npm run build` and import the emitted ESM. That keeps the
 * rig dependency-free and free of ambient Node type packages.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { LlmWikiClient, LlmWikiApiError } from "../dist/src/index.js";
import {
  LlmWikiClientWithEmbeddingJobs,
  withEmbeddingJobs,
} from "../dist/src/embed.js";

const BASE = "http://127.0.0.1:8080";

/** A fetch stub that records every request and replays queued responses. */
function recordingFetch(responses) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({
      url,
      method: init.method,
      headers: init.headers,
      body: init.body === undefined ? undefined : JSON.parse(init.body),
    });
    const next = responses.shift();
    if (!next) throw new Error(`unexpected request: ${init.method} ${url}`);
    return new Response(next.body === undefined ? "" : JSON.stringify(next.body), {
      status: next.status ?? 200,
      headers: next.headers ?? { "content-type": "application/json" },
    });
  };
  return { fetchImpl, calls };
}

test("startEmbedJob POSTs to /v1/embed/jobs with model, batch and the idempotency key", async () => {
  const { fetchImpl, calls } = recordingFetch([
    { status: 202, body: { job_id: "job-1", status: "queued" } },
  ]);
  const wiki = new LlmWikiClientWithEmbeddingJobs({
    baseUrl: BASE,
    apiKey: "secret",
    fetch: fetchImpl,
  });

  const ack = await wiki.startEmbedJob("bge-m3", { batch: 32, idempotencyKey: "nightly-1" });

  assert.deepEqual(ack, { job_id: "job-1", status: "queued" });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].method, "POST");
  assert.equal(calls[0].url, `${BASE}/v1/embed/jobs`);
  assert.deepEqual(calls[0].body, { model: "bge-m3", batch: 32 });
  assert.equal(calls[0].headers["idempotency-key"], "nightly-1");
  assert.equal(calls[0].headers.authorization, "Bearer secret");
  assert.equal(calls[0].headers["content-type"], "application/json");
});

test("startEmbedJob omits the idempotency key and empty knobs when not asked", async () => {
  const { fetchImpl, calls } = recordingFetch([
    { status: 202, body: { job_id: "job-2", status: "queued" } },
  ]);
  const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: BASE, fetch: fetchImpl });

  await wiki.startEmbedJob();

  assert.equal(calls[0].headers["idempotency-key"], undefined);
  assert.equal(calls[0].headers.authorization, undefined);
  // undefined knobs must not serialize into keys the server would parse.
  assert.deepEqual(calls[0].body, {});
});

test("embed POSTs the synchronous /v1/embed and parses the backfill report", async () => {
  const { fetchImpl, calls } = recordingFetch([
    {
      body: {
        protocol_version: 1,
        request_id: "req-1",
        generation: "gen-1",
        model: "bge-m3",
        total_sections: 100,
        covered_before: 90,
        embedded: 10,
      },
    },
  ]);
  const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: BASE, fetch: fetchImpl });

  const report = await wiki.embed("bge-m3", { batch: 16 });

  assert.equal(calls[0].method, "POST");
  assert.equal(calls[0].url, `${BASE}/v1/embed`);
  assert.deepEqual(calls[0].body, { model: "bge-m3", batch: 16 });
  assert.equal(report.total_sections, 100);
  assert.equal(report.covered_before, 90);
  assert.equal(report.embedded, 10);
  assert.equal(report.generation, "gen-1");
});

test("job() surfaces the embed progress counters", async () => {
  const { fetchImpl, calls } = recordingFetch([
    {
      body: {
        protocol_version: 1,
        request_id: "req-2",
        job_id: "job-3",
        kind: "embed",
        status: "RUNNING",
        phase: "EMBEDDING",
        build_id: null,
        failure_code: null,
        retryable: false,
        error: null,
        created_at: "2026-10-14T00:00:00Z",
        started_at: "2026-10-14T00:00:01Z",
        finished_at: null,
        progress: { total: 100, covered_before: 90, embedded: 4 },
      },
    },
  ]);
  const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: BASE, fetch: fetchImpl });

  const job = await wiki.job("job-3");

  assert.equal(calls[0].url, `${BASE}/v1/jobs/job-3`);
  assert.equal(job.kind, "embed");
  assert.equal(job.phase, "EMBEDDING");
  assert.deepEqual(job.progress, { total: 100, covered_before: 90, embedded: 4 });
});

test("cancelJob POSTs the cooperative cancel endpoint", async () => {
  const { fetchImpl, calls } = recordingFetch([
    { body: { job_id: "job-4", status: "CANCELLED" } },
  ]);
  const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: BASE, fetch: fetchImpl });

  const cancelled = await wiki.cancelJob("job-4");

  assert.equal(calls[0].method, "POST");
  assert.equal(calls[0].url, `${BASE}/v1/jobs/job-4/cancel`);
  assert.equal(cancelled.status, "CANCELLED");
});

test("a 409 embed_already_running surfaces as a coded error with the request id", async () => {
  const { fetchImpl } = recordingFetch([
    {
      status: 409,
      headers: { "content-type": "application/json", "x-request-id": "req-9" },
      body: {
        protocol_version: 1,
        request_id: "req-9",
        error: {
          code: "embed_already_running",
          message: "an embed job is already running; one backfill at a time",
        },
      },
    },
  ]);
  const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: BASE, fetch: fetchImpl });

  await assert.rejects(
    () => wiki.startEmbedJob("bge-m3"),
    (err) => {
      assert.ok(err instanceof LlmWikiApiError);
      assert.equal(err.status, 409);
      assert.equal(err.code, "embed_already_running");
      assert.equal(err.requestId, "req-9");
      return true;
    },
  );
});

test("one client = one workspace: the mixin still speaks protocol v1 only", async () => {
  const { fetchImpl } = recordingFetch([
    {
      status: 200,
      body: { protocol_version: 2, request_id: "req-10", hits: [], truncated: false },
    },
  ]);
  const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: BASE, fetch: fetchImpl });

  await assert.rejects(
    () => wiki.search("anything"),
    (err) => err instanceof LlmWikiApiError && err.code === "protocol_mismatch",
  );
});

test("the extended client IS a protocol-v1 client (additive surface, no fork)", () => {
  // Each withEmbeddingJobs(Base) call returns a fresh class, so the property
  // worth pinning is inheritance, not class identity: whatever a consumer
  // builds, it is still a v1 LlmWikiClient with the embed surface added.
  const Extended = withEmbeddingJobs(LlmWikiClient);
  const wiki = new Extended({ baseUrl: BASE, fetch: async () => new Response("{}") });

  assert.ok(wiki instanceof LlmWikiClient, "extending must not fork the v1 client");
  assert.ok(
    new LlmWikiClientWithEmbeddingJobs({ baseUrl: BASE, fetch: async () => new Response("{}") }) instanceof
      LlmWikiClient,
  );
  assert.equal(typeof wiki.status, "function", "inherited v1 surface");
  assert.equal(typeof wiki.job, "function", "inherited v1 surface");
  assert.equal(typeof wiki.startEmbedJob, "function", "added surface");
  assert.equal(typeof wiki.embed, "function", "added surface");
});
