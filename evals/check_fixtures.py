#!/usr/bin/env python3
"""Read-only integrity checks for the §37.3 eval fixtures.

Verifies that dataset.yaml spans appear verbatim in the corpus, that all
referenced docs exist, and that the fixture meets the V0.1 minimums
(≥30 docs, ≥20 questions, ≥2 mdx, ≥2 CJK docs). Exits non-zero on any
violation. Run from the repo root:

    python evals/check_fixtures.py

No files are written.
"""

import sys
from pathlib import Path

MIN_DOCS = 30
MIN_QUESTIONS = 20
MIN_MDX = 2
MIN_CJK = 2

EVALS = Path(__file__).resolve().parent
CORPUS = EVALS / "corpus"


def parse_simple_yaml(path):
    """Minimal parser for the fixture files' fixed shape: 0-3 space indented
    mappings/lists with string or int scalars. Raises on anything else."""
    def scalar(raw):
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if raw.isdigit():
            return int(raw)
        return raw

    root = {}
    stack = [(-1, root)]
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.rstrip("\n")
            if not stripped.strip() or stripped.strip().startswith("#"):
                continue
            indent = len(stripped) - len(stripped.lstrip(" "))
            content = stripped.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if content.startswith("- "):
                rest = content[2:].strip()
                if ":" not in rest:
                    # scalar list item (e.g. `- auth/authentication.md`)
                    if not isinstance(parent, list):
                        raise ValueError(f"line {lineno}: scalar item in mapping")
                    parent.append(scalar(rest))
                    continue
                item = {}
                parent.append(item)
                stack.append((indent, item))
                parent = item
                content = rest
            key, _, value = content.partition(":")
            key = key.strip()
            value = value.strip()
            if value == "":
                child = [] if _next_is_list_item(path, lineno) else {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = scalar(value)
    return root


def _next_is_list_item(path, lineno):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i <= lineno:
                continue
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return s.startswith("- ")
    return False


def main():
    problems = []

    dataset = parse_simple_yaml(EVALS / "dataset.yaml")
    questions = parse_simple_yaml(EVALS / "questions.yaml")
    pages = parse_simple_yaml(EVALS / "expected" / "pages.yaml")

    docs = dataset["docs"]
    mdx_count = sum(1 for d in docs if d["path"].endswith(".mdx"))
    cjk_count = sum(1 for d in docs if str(d.get("language", "")).startswith("zh"))

    if len(docs) < MIN_DOCS:
        problems.append(f"corpus has {len(docs)} docs, need >= {MIN_DOCS}")
    if mdx_count < MIN_MDX:
        problems.append(f"corpus has {mdx_count} .mdx docs, need >= {MIN_MDX}")
    if cjk_count < MIN_CJK:
        problems.append(f"corpus has {cjk_count} CJK docs, need >= {MIN_CJK}")
    if len(questions["questions"]) < MIN_QUESTIONS:
        problems.append(
            f"{len(questions['questions'])} questions, need >= {MIN_QUESTIONS}"
        )

    seen_ids = set()
    fact_count = 0
    high_count = 0
    annotated_paths = set()
    for doc in docs:
        rel = doc["path"]
        annotated_paths.add(rel)
        body_path = CORPUS / rel
        if not body_path.is_file():
            problems.append(f"annotated doc missing from corpus: {rel}")
            continue
        body = body_path.read_text(encoding="utf-8")
        for fact in doc["facts"]:
            fact_count += 1
            if fact["id"] in seen_ids:
                problems.append(f"duplicate fact id: {fact['id']}")
            seen_ids.add(fact["id"])
            if fact["importance"] == "high":
                high_count += 1
            if fact["span"] not in body:
                problems.append(f"span not found verbatim in {rel}: {fact['span']!r}")

    known = annotated_paths
    for q in questions["questions"]:
        for src in q["expected_sources"]:
            if src not in known:
                problems.append(f"{q['id']}: unknown expected source {src}")
    for page in pages["pages"]:
        sources = page["sources"]
        for src in sources:
            if src not in known:
                problems.append(f"page {page['title']!r}: unknown source {src}")
        if page["min_sources_merged"] > len(sources):
            problems.append(
                f"page {page['title']!r}: min_sources_merged > listed sources"
            )

    print(
        f"fixtures: {len(docs)} docs ({mdx_count} mdx, {cjk_count} cjk), "
        f"{fact_count} facts ({high_count} high), "
        f"{len(questions['questions'])} questions, {len(pages['pages'])} expected pages"
    )
    if problems:
        for p in problems:
            print(f"FAIL {p}")
        return 1
    print("fixtures: all integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
