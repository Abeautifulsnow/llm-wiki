<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

This project is managed by Trellis. The working knowledge you need lives under `.trellis/`:

- `.trellis/workflow.md` — development phases, when to create tasks, skill routing
- `.trellis/spec/` — package- and layer-scoped coding guidelines (read before writing code in a given layer)
- `.trellis/workspace/` — per-developer journals and session traces
- `.trellis/tasks/` — active and archived tasks (PRDs, research, jsonl context)

If a Trellis command is available on your platform (e.g. `/trellis:finish-work`, `/trellis:continue`), prefer it over manual steps. Not every platform exposes every command.

If you're using Codex or another agent-capable tool, additional project-scoped helpers may live in:
- `.agents/skills/` — reusable Trellis skills
- `.codex/agents/` — optional custom subagents

Managed by Trellis. Edits outside this block are preserved; edits inside may be overwritten by a future `trellis update`.

<!-- TRELLIS:END -->

# Agent Tool Call Rules

## Codebase Analysis & Navigation (CodeGraph)

This repository has been indexed using CodeGraph.

When answering architecture questions, looking up symbol definitions, searching for callers/callees, or performing change impact analysis:

1. **Always use the CodeGraph MCP tool first** (`mcp__codegraph__codegraph_explore`).
2. **Do NOT crawl or scan files manually** using standard `grep`, `glob`, or sequential `read_file` unless you need to view exact line edits for a specific file.
3. Trust the graph-backed context provided by CodeGraph to minimize unnecessary tool calls and context usage.
