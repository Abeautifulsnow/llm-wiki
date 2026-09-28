# Eval Fixtures (PRD §37.3)

Deterministic evaluation fixtures for the Knowledge Compiler. CI runs all
gates with `FakeLlmProvider` (crate `llm-wiki-evals`); real-LLM evaluation is
an explicit, separately triggered addition, never part of CI.

## Layout

| path | purpose |
|---|---|
| `corpus/` | 33 Markdown/MDX source docs (synthetic "Nimbus" platform docs; includes CJK and `.mdx` files) |
| `dataset.yaml` | per-doc annotated source facts: `id`, verbatim `span`, `importance` (`high` facts form the Source Coverage denominator) |
| `questions.yaml` | 26 retrieval questions with `expected_sources` (fixture-only in V0.1; scored from V0.2 search onward) |
| `expected/pages.yaml` | required subset of the compiled wiki: pages that must exist and the minimum number of distinct sources each must cite (Cross-document Synthesis gate) |
| `check_fixtures.py` | read-only integrity validator (spans verbatim, references resolve, minimums met) |

## Annotation format

- `dataset.yaml`: one entry per corpus doc; every fact is a **verbatim
  sentence** from that doc (`span`). `importance: high` marks facts that the
  Source Coverage gate counts; `normal` facts are annotation-only context.
- `questions.yaml`: `expected_sources` lists the corpus docs that must supply
  the supporting facts for the question.
- `expected/pages.yaml`: this is the **required subset**, not the full page
  inventory — gates assert presence and citation breadth, nothing else.

## Metric denominators (V0.1)

| gate | numerator / denominator | threshold |
|---|---|---|
| Source Coverage | annotated `high` facts represented by a knowledge claim / all `high` facts | ≥ 90% |
| Citation Correctness | citations whose range is inside the source, hash matches, digest consistent / all citations on expected pages | ≥ 95% and **zero** invalid range/hash |
| Hallucination Rate | falsifiable claims with no source backing / all claims on expected pages | ≤ 5% |
| Cross-document Synthesis | expected pages citing ≥ `min_sources_merged` distinct sources / expected pages | all must pass; ≥1 page merges ≥3 sources |
| Rebuild Determinism | new LLM requests on second identical build; structured manifest (page IDs, citation mapping, links) compared field-by-field | 0 new requests, manifest identical |

## Threshold changes are a product contract change

Any threshold change must be recorded here with a version note:

| version | date | change |
|---|---|---|
| 1 | 2026-09-28 | initial V0.1 thresholds (§37.3) |

## Usage

```bash
python evals/check_fixtures.py        # fixture integrity (no writes)
cargo test -p llm-wiki-evals          # gate tests (FakeLlmProvider, CI-safe)
```
