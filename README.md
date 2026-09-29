# llm-wiki

> LLM-Wiki is a knowledge compiler that turns evolving documentation into a persistent, grounded and incrementally maintained Wiki for humans and AI agents.

```text
Documents are not merely indexed.
They are compiled into maintainable knowledge.
```

Not another RAG framework: sources stay untouched under `docs/`, and the wiki
is a derived, fully cited artifact — every claim carries a source file,
heading path, byte range and content digest that the compiler (never the
model) writes down.

## How it works

```text
Raw Sources → Parse/Normalize → Knowledge Extraction (LLM, structured + verified)
→ Stable Knowledge Registry → Hierarchical Planning → Page Compilation
→ Atomic Publish (generations/ + current.json) → FTS Search / Graph / lint
```

Key properties (see `docs/llm-wiki-prd.md` for the full spec):

- **Grounded** — claims are verified against source ranges; unverifiable
  output lands in an audited `rejected_claims` table, never in your wiki.
- **Incremental** — rebuilds touch only the pages the diff proves affected;
  anything that cannot be proven safe stops as `REPLAN_REQUIRED` instead of
  silently mixing old and new plans.
- **Reproducible** — identical inputs make zero new LLM requests (validated
  responses are cached per fingerprint).
- **Atomic** — a failed build never damages the currently visible wiki.
- **Chinese-aware search** — FTS5 with a shared CJK unigram/bigram +
  Latin-word analyzer; `search.graph = true` adds one-hop related pages
  from the built-in Wiki Graph.

## Quick start

Requires the Rust toolchain and an OpenAI-compatible LLM endpoint
(vLLM / Ollama / OpenRouter / the official API all work).

```bash
cargo build --release

export LLM_WIKI_API_KEY=sk-…        # name configurable via config.api_key_env

llm-wiki init                       # writes .llm-wiki/config.toml (source root ./docs)
llm-wiki scan ./docs                # scan + manifest diff, no LLM calls
llm-wiki build                      # full compile (first build spends real tokens)
llm-wiki search "单点登录"           # CJK full-text search over the compiled wiki
llm-wiki lint                       # broken links, stale citations, hand edits, …
llm-wiki status                     # sources, builds, active generation
llm-wiki doctor                     # config / db / publish / FTS5 availability
```

Output lives under `wiki/generations/{build_id}/*.md` with a `current.json`
pointer — plain Markdown any LLM or agent can consume directly.

### Changing model, prompt or config

The build fingerprint covers model + prompt + schema + parser + config
versions. A drift stops the incremental build with `REPLAN_REQUIRED`
(exit code 7) instead of rewriting pages behind your back. Review what
changed, then opt in explicitly:

```bash
llm-wiki replan --dry-run           # audit the plan diff + cost estimate, no writes
llm-wiki replan                     # execute: stable page IDs survive merges/splits
```

### What to expect on cost

- The **first build** compiles everything: real tokens, real time.
- **Rebuilds** only re-analyze changed sources and recompile affected pages;
  unchanged pages are carried over verbatim.
- **Second build with no changes** makes zero LLM requests (cache).

Model quality matters: extraction demands strict JSON and precise evidence
ranges. Small local models may trip the rejected-claim safety gate — a
mid-tier or larger model is recommended until measured otherwise. Thinking
models (deepseek-style) burn reasoning tokens from the request output budget:
raise `[llm] max_output_tokens` (e.g. 32768) or analysis fails with
`NO_JSON` before any visible output.

## Development

```bash
cargo test --workspace              # 250+ tests, FakeLlmProvider only
cargo clippy --workspace --all-targets -- -D warnings
python3 evals/check_fixtures.py     # eval corpus integrity (read-only)
```

Real-LLM E2E is `#[ignore]`-gated and never runs in CI:

```bash
LLM_WIKI_E2E_API_KEY=… LLM_WIKI_E2E_BASE_URL=… \
  cargo test -p llm-wiki-compiler --test golden -- --ignored
```

Roadmap: V0.3 vector retrieval, V0.4 HTTP service, V0.5 hybrid retrieval +
query. The current milestone is V0.2 (incremental build, explicit replan,
full-text search, wiki graph).
