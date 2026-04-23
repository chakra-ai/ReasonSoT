# ReasonSoT

**Low-latency dual-process reasoning engine for voice interview agents.**

ReasonSoT is a Python library that conducts adaptive, intelligent interviews over voice. It combines a **zero-cost heuristic router** with a **System 1 / System 2** model split (Haiku / Sonnet) and a family of single-prompt reasoning modes (CoT, Matrix of Thought, Domain-Specialized Tree, Speculative CoT) so that most turns avoid the latency of deep reasoning entirely, and deep turns only spend as many thinking tokens as they actually need.

```
User utterance ──► Heuristic Router ──► System 1 (Haiku)                     ──► streamed reply
                        │                   or
                        ├────────────────► System 2 (Sonnet + extended thinking)
                        │                   with CoT / MoT / DST
                        └────────────────► Speculative (S1 draft → S2 verify)
```

## Why this exists

Voice interviews have a hard latency budget — time-to-first-token above ~600 ms feels unnatural. But not every turn needs deep reasoning: "hello" and "can you repeat that?" cost the same tokens as "walk me through how you'd design a distributed rate limiter" under a naive single-model setup. ReasonSoT splits the traffic:

- **Simple turns** (greetings, acknowledgments, transitions) go through a Haiku-only fast path with no extended thinking.
- **Deep turns** (system-design probes, trade-off questions) go through Sonnet with a reasoning mode selected from user-input signals and a thinking budget sized to the complexity of the question.
- **Borderline turns** use **speculative CoT**: Haiku drafts an answer with a self-assessed confidence score, and Sonnet only verifies if confidence is low.

The routing decision is made in microseconds with regex patterns, technical-term density, and conversation-state signals — no LLM call is needed to pick the path.

## Quick start

```bash
# Python 3.12+
make install                     # pip install -e ".[dev]"
cp .env.example .env             # then edit with your ANTHROPIC_API_KEY
make demo                        # interactive CLI interview
make demo ARGS="--verbose"       # per-turn latency + routing breakdown
```

Or run a specific persona:

```bash
python demo.py --persona behavioral_interviewer --verbose
```

## Commands

| Command | What it does |
|---|---|
| `make install` | Install package in editable mode with dev deps |
| `make test` | Unit tests only (`pytest -m "not integration"`) |
| `make test-integration` | Tests that hit the real Anthropic API |
| `make demo` | Interactive CLI interview |
| `make bench` | Run benchmarks vs. baselines, write JSON to `benchmarks/results/` |
| `make lint` | Minimal `py_compile` check |

Run a single test: `pytest tests/test_router.py::test_simple_pattern_routes_s1 -v`.

## Docs

Deep documentation lives in [`docs/`](./docs):

- [Overview](./docs/overview.md) — the "why" and overall approach
- [Architecture](./docs/architecture.md) — components and module map
- [Turn workflow](./docs/workflow.md) — end-to-end per-turn sequence
- [Design: Routing](./docs/design/routing.md) — heuristic router deep dive
- [Design: Reasoning modes](./docs/design/reasoning-modes.md) — CoT / MoT / DST / Speculative
- [Design: Knowledge graph](./docs/design/knowledge-graph.md) — in-memory KG for coverage tracking
- [Design: Prefix caching](./docs/design/prefix-caching.md) — cache strategy for sub-500ms TTFT

`CLAUDE.md` is a short, dense orientation for AI coding assistants working in this repo.

## Layout

```
reason_sot/
  core/           # Routing, System 1, System 2, reasoning modes (CoT/MoT/DST), speculative, early-exit
  interview/      # Session engine, follow-up classifier, knowledge graph
  llm/            # Anthropic client, model registry, prefix-cache builder
  persona/        # Pydantic persona profiles, YAML definitions, persona manager
  scoring/        # Reasoning-quality metrics, latency profiler, benchmark harness
benchmarks/       # Scenario definitions and benchmark runner
tests/            # Pytest suite with MockLLMClient
demo.py           # Interactive CLI
config.py         # Class-based config (Dev / Prod / Test)
```

## License

MIT (pending — see `LICENSE` if added).
