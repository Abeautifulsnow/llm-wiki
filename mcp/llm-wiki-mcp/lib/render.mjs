/**
 * Response rendering: protocol-v1 JSON in, model-facing Markdown out.
 *
 * These are pure functions so both transports produce byte-identical tool
 * results — the transport must never change what the model sees.
 */

export function clampText(text, limit) {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}\n\n…[truncated ${text.length - limit} more characters]`;
}

/** Explains a per-side degradation object; "" when everything served. */
export function degradedNote(degraded) {
  if (!degraded || typeof degraded !== "object") return "";
  const problems = Object.entries(degraded).filter(([, status]) => status !== "served");
  if (problems.length === 0) return "";
  const detail = problems.map(([side, status]) => `${side}=${status}`).join(", ");
  return `retrieval degraded (${detail}) — this is NOT proof the knowledge base is empty.`;
}

/** Page path for wiki chunks; `file:start-end` for raw-source chunks. */
function chunkLocation(chunk) {
  if (chunk?.source_ref) {
    const ref = chunk.source_ref;
    const file = ref.file_path ?? ref.source_id ?? "(unknown source)";
    const range =
      Number.isInteger(ref.range_start) && Number.isInteger(ref.range_end)
        ? `:${ref.range_start}-${ref.range_end}`
        : "";
    return `${file}${range}`;
  }
  const parts = [chunk?.slug, ...(Array.isArray(chunk?.heading_path) ? chunk.heading_path : [])].filter(Boolean);
  return parts.join(" › ");
}

export function renderContext(data) {
  const lines = [];
  const note = degradedNote(data.degraded);
  if (note) lines.push(`> ⚠ ${note}`, "");

  lines.push(
    `generation: ${data.generation ?? "(none)"} · mode: ${data.served_mode ?? "wiki"} · ` +
      `est_tokens: ${data.estimated_tokens ?? "?"} · truncated: ${data.truncated ? "yes" : "no"}`,
    "",
  );

  const chunks = Array.isArray(data.chunks) ? data.chunks : [];
  if (chunks.length === 0) {
    lines.push(
      "No chunks matched. Try different wording (keep entity names and identifiers verbatim), " +
        "or call `wiki_status` to check that a wiki is published.",
    );
    return lines.join("\n");
  }

  chunks.forEach((chunk, index) => {
    lines.push(`### ${index + 1}. ${chunk.title || chunk.slug || "(untitled)"}`);
    const location = chunkLocation(chunk);
    if (location) lines.push(`- location: ${location}`);
    const kind = chunk.evidence_kind ?? "wiki";
    const sources =
      Array.isArray(chunk.sources) && chunk.sources.length > 0 ? ` · sources: ${chunk.sources.join(", ")}` : "";
    lines.push(`- evidence: ${kind}${sources}`);
    if (Number.isFinite(chunk.score)) lines.push(`- score: ${chunk.score.toFixed(3)}`);
    lines.push("", String(chunk.snippet ?? "").trim() || "(empty snippet)", "");
  });

  const neighbors = Array.isArray(data.neighbors) ? data.neighbors : [];
  if (neighbors.length > 0) {
    lines.push("Related (wiki graph):");
    for (const neighbor of neighbors) {
      lines.push(`- ${neighbor.label ?? neighbor.node_id} (${neighbor.relation ?? "related"}) ← ${neighbor.from_slug}`);
    }
    lines.push("");
  }

  lines.push(
    "Cite the `location` of every claim you take from these chunks. " +
      "If the chunks do not answer the question, say so instead of guessing.",
  );
  return lines.join("\n");
}

export function renderSearch(data) {
  const lines = [];
  const note = degradedNote(data.served);
  if (note) lines.push(`> ⚠ ${note}`, "");
  lines.push(
    `generation: ${data.generation ?? "(none)"} · mode: ${data.source_mode ?? "wiki"} · ` +
      `truncated: ${data.truncated ? "yes" : "no"}`,
    "",
  );

  const hits = Array.isArray(data.hits) ? data.hits : [];
  if (hits.length === 0) {
    lines.push("No hits. Try different wording, or call `wiki_status` to check the published generation.");
    return lines.join("\n");
  }

  hits.forEach((hit, index) => {
    lines.push(`${index + 1}. ${hit.title || hit.slug || "(untitled)"}`);
    const location = chunkLocation(hit);
    if (location) lines.push(`   location: ${location}`);
    lines.push(`   kind: ${hit.evidence_kind ?? "wiki"}`);
    const snippet = String(hit.snippet ?? "").trim().replace(/\s+/g, " ");
    if (snippet) lines.push(`   ${snippet}`);
    lines.push("");
  });

  lines.push("These are page/section hits, not answers. Use `wiki_context` to pull the full snippets.");
  return lines.join("\n");
}

export function renderStatus(data) {
  const lines = [
    `workspace: ${data.workspace ?? "(unknown)"}`,
    `sources: ${data.sources ?? "?"}`,
    `active generation: ${data.generation ?? data.active_build_id ?? "(none)"}`,
  ];
  if (data.latest_build) {
    lines.push(
      `latest build: ${data.latest_build.status ?? "?"} (${data.latest_build.build_id ?? "?"}, ` +
        `started ${data.latest_build.started_at ?? "?"})`,
    );
  } else {
    lines.push("latest build: (none)");
  }
  if (data.versions) {
    lines.push(
      `versions: binary ${data.versions.binary ?? "?"} · compiler ${data.versions.generation_compiler ?? "?"} · ` +
        `schema ${data.versions.db_schema ?? "?"}`,
    );
    if (data.versions.generation_stale === true) {
      lines.push(
        "⚠ generation_stale = true: the published wiki predates this binary. Run `llm-wiki build` " +
          "(or expose `wiki_build`) before trusting answers.",
      );
    }
  }
  if (data.jobs) {
    const jobs = Object.entries(data.jobs)
      .filter(([, count]) => Number(count) > 0)
      .map(([status, count]) => `${status}=${count}`)
      .join(" ");
    lines.push(`jobs: ${jobs || "idle"}`);
  }
  if (!data.generation && !data.active_build_id) {
    lines.push("", "Nothing is published yet: run `llm-wiki build` in the workspace, then `llm-wiki serve`.");
  }
  return lines.join("\n");
}

export function renderPage(data) {
  const lines = [
    `# ${data.title ?? data.slug}`,
    "",
    `- slug: ${data.slug}`,
    `- page_id: ${data.page_id}`,
    `- category: ${data.category ?? "(none)"}`,
    `- language: ${data.language ?? "?"}`,
    "",
    "---",
    "",
    String(data.content ?? "").trim(),
  ];

  const citations = Array.isArray(data.citations) ? data.citations : [];
  if (citations.length > 0) {
    lines.push("", "---", "", `Citations (${citations.length}):`);
    for (const citation of citations.slice(0, 40)) {
      const range = Array.isArray(citation.range) ? `:${citation.range[0]}-${citation.range[1]}` : "";
      const heading = Array.isArray(citation.heading_path) ? citation.heading_path.join(" › ") : "";
      lines.push(
        `- ${citation.source_id}${range}${heading ? ` · ${heading}` : ""} · ` +
          `digest ${String(citation.evidence_digest ?? "").slice(0, 12)}`,
      );
    }
    if (citations.length > 40) lines.push(`- … ${citations.length - 40} more`);
  }

  const links = Array.isArray(data.links) ? data.links : [];
  if (links.length > 0) {
    lines.push("", "Linked pages:");
    for (const link of links) lines.push(`- ${link.target_title ?? link.to_page_id}`);
  }
  return lines.join("\n");
}

export function renderAnswer(data) {
  const lines = [String(data.answer ?? "").trim()];
  const citations = Array.isArray(data.citations) ? data.citations : [];
  if (citations.length > 0) {
    lines.push("", "Citations:");
    for (const citation of citations.slice(0, 40)) {
      const range = Array.isArray(citation.range) ? `:${citation.range[0]}-${citation.range[1]}` : "";
      lines.push(`- ${citation.source ?? citation.source_id ?? "?"}${range}`);
    }
  }
  if (data.insight_id) lines.push("", `Persisted as insight ${data.insight_id}.`);
  return lines.join("\n");
}

export function renderJob(data) {
  const lines = [
    `job: ${data.job_id ?? "?"} (${data.kind ?? "?"})`,
    `status: ${data.status ?? "?"}${data.phase ? ` · phase ${data.phase}` : ""}`,
  ];
  if (data.failure_code) lines.push(`failure_code: ${data.failure_code}`);
  if (data.error) lines.push(`error: ${data.error}`);
  if (data.progress && typeof data.progress === "object") {
    lines.push(`progress: ${Object.entries(data.progress).map(([key, value]) => `${key}=${value}`).join(" ")}`);
  }
  lines.push(
    `created: ${data.created_at ?? "?"} · started: ${data.started_at ?? "-"} · finished: ${data.finished_at ?? "-"}`,
  );
  return lines.join("\n");
}
