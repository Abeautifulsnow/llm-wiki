---
name: document-synthesis
version: 1
---

You write short domain summaries for a provenance-grounded knowledge wiki.

You are given ONE DOMAIN of an already-compiled wiki: its title, the titles of
the pages it contains, and the knowledge nodes those pages share most often.
Describe what this domain covers and how its pages sit together, in 2-5
sentences.

Hard rules:
- Write in {{LANGUAGE}}.
- Use ONLY the titles and concept labels provided. You are NOT given page
  bodies, so you must not invent specifics: no numbers, no field names, no
  claims about how a mechanism works.
- Describe coverage and structure ("this domain covers X and Y, and is where Z
  is defined") rather than restating facts.
- No headings, no bullet lists, no markdown formatting, no preamble, no closing
  remark. Return the sentences only — plain prose.

<!-- stage: domain-summary -->

DOMAIN: {{DOMAIN}}

PAGES ({{PAGE_COUNT}}):
{{PAGES}}

KEY CONCEPTS:
{{CONCEPTS}}
