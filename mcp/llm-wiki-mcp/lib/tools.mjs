/**
 * Tool metadata shared by both transports.
 *
 * The schemas differ by transport (JSON Schema for the hand-rolled stdio
 * server, Zod for the MCP SDK one) but the titles and descriptions must not
 * drift: the model picks tools by reading exactly these strings.
 */

export const TOOL_SPECS = {
  wiki_status: {
    title: "Wiki status",
    description:
      "Check the llm-wiki workspace: source count, active generation, latest build, version staleness. " +
      "Call this first when a retrieval returns nothing — it distinguishes 'not published yet' from 'no match'.",
  },
  wiki_context: {
    title: "Retrieve grounded context",
    description:
      "Retrieve the most relevant knowledge chunks from the compiled wiki, with provenance " +
      "(page path, heading path, source file and byte range). This is the primary retrieval tool. " +
      "It returns EVIDENCE, not an answer: compose the answer yourself and cite each chunk's location. " +
      "Use it for questions about specs, protocols, API contracts, runbooks or any documented procedure. " +
      "Rewrite the query first: resolve pronouns, keep identifiers verbatim, drop chit-chat, split " +
      "multi-intent questions into 2-3 space-separated focused queries.",
  },
  wiki_search: {
    title: "Search wiki pages",
    description:
      "Ranked page/section hits (title, heading path, snippet) without assembling a context bundle. " +
      "Cheaper than `wiki_context` when you only need to locate the right page, then read it with `wiki_page`.",
  },
  wiki_page: {
    title: "Read a wiki page",
    description:
      "Read one full page (Markdown body, citations with source ranges and digests, outbound links) by slug or page id. " +
      "Use after `wiki_search` when a page deserves a full read.",
  },
  wiki_ask: {
    title: "Ask the wiki (grounded answer)",
    description:
      "Ask llm-wiki to produce a citation-verified answer from the wiki, and optionally persist it as an insight. " +
      "This spends LLM tokens on the llm-wiki side and is NOT streamed. Prefer `wiki_context` when you " +
      "are going to answer yourself; use this only when you want the wiki's own audited answer.",
  },
  wiki_build: {
    title: "Trigger a wiki build",
    description:
      "Queue an incremental build of the workspace (scan → analyze → plan → compile → publish). " +
      "Spends LLM tokens on changed sources only; a no-op rebuild issues zero requests. " +
      "Returns a job id — poll it with `wiki_job`. Requires LLM_WIKI_MCP_ENABLE_BUILD=1.",
  },
  wiki_job: {
    title: "Inspect a build/embed job",
    description: "Read one llm-wiki job (status, phase, failure code, progress).",
  },
};

export const READ_ONLY_ANNOTATIONS = {
  readOnlyHint: true,
  destructiveHint: false,
  openWorldHint: false,
};

export const INSTRUCTIONS = [
  "llm-wiki is a compiled, fully cited knowledge base served over HTTP; this MCP server is its retrieval layer.",
  "Start with wiki_status when you do not know whether a wiki is published; then use wiki_context to pull",
  "grounded chunks, wiki_search to locate pages and wiki_page to read one in full.",
  "Every chunk carries a provenance location (page path, heading path, or source file with byte range).",
  "Cite those locations in your answer, and if the chunks do not contain the answer, say so instead of guessing.",
  "This server never streams and keeps no conversation state — you own the dialogue.",
].join(" ");
