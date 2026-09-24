# llm-wiki-core

Domain models and core abstractions for LLM-Wiki (PRD §7.1).

This crate must never depend on Axum, Clap, any OpenAI SDK, a SQLite driver or a
vector/graph database (PRD §7.1). Runtime adapters live in their own crates and
depend on this one, never the reverse.
