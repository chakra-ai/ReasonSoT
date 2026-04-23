# ReasonSoT Documentation

This folder documents the design, architecture, and per-component behavior of ReasonSoT.

## HTML version

For a browser-friendly read with diagrams rendered inline, open [`docs/html/index.html`](./html/index.html). The HTML pages cover the approach, solution architecture, and workflow with Mermaid diagrams auto-rendered via CDN. Open the file directly in a browser — no build step.

## Reading order

If you're new to the project, read in this order:

1. **[Overview](./overview.md)** — the problem, the approach, the key ideas. ~5 min read.
2. **[Architecture](./architecture.md)** — components, module map, data flow at the system level.
3. **[Turn workflow](./workflow.md)** — what happens end-to-end for a single user utterance, with sequence diagrams.

Then dip into the design notes for specific subsystems:

- **[Routing](./design/routing.md)** — how `core/router.py` decides between System 1, System 2, and speculative paths without an LLM call.
- **[Reasoning modes](./design/reasoning-modes.md)** — CoT, Matrix of Thought, Domain-Specialized Tree, and Speculative CoT.
- **[Knowledge graph](./design/knowledge-graph.md)** — the in-memory KG that tracks topic coverage and informs routing.
- **[Prefix caching](./design/prefix-caching.md)** — how we hit the 90%+ cache-read rate that makes sub-500ms TTFT possible.

## Diagram conventions

All diagrams are written in **Mermaid**, which GitHub renders natively. If you're viewing these files in an editor that doesn't render Mermaid, paste them into <https://mermaid.live> or the GitHub web UI.

Shapes used consistently across docs:

- **Rectangles** — deterministic Python code (router, KG updates, cache builders)
- **Rounded rectangles / stadiums** — LLM calls (System 1 or System 2)
- **Diamonds** — decision points
- **Cylinders** — persistent state (session, conversation, KG)

## Keeping docs fresh

If you change behavior that's documented here (routing thresholds, reasoning-mode prompts, cache block layout), update the relevant doc in the same PR. Code that drifts from these docs is harder to reason about than code with no docs at all.
