# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install (editable + dev deps): `make install` (runs `pip install -e ".[dev]"`, requires Python >=3.12).

Tests:
- `make test` — unit tests only (`pytest -m "not integration"`)
- `make test-integration` — tests that hit the real Anthropic API (require `ANTHROPIC_API_KEY`)
- Single test: `pytest tests/test_router.py::test_simple_pattern_routes_s1 -v`
- Async tests use `asyncio_mode = "auto"` (pyproject.toml) — don't add `@pytest.mark.asyncio` decorators.

Run the demo: `make demo` or `python demo.py [--persona <name>] [--verbose]`. Available personas are discovered from `reason_sot/persona/definitions/*.yaml`.

Benchmarks: `make bench` (runs `benchmarks/run_benchmarks.py`, writes JSON to `benchmarks/results/`). Scenario subset: `python benchmarks/run_benchmarks.py --scenario technical_python --configs baseline,reason_sot`.

Lint (minimal): `make lint` (just `py_compile` — no real linter configured).

Env setup: copy `.env.example` → `.env`, set `ANTHROPIC_API_KEY`. `REASON_SOT_ENV` selects `DevelopmentConfig`/`ProductionConfig`/`TestConfig` via `config.get_config()`. `TestConfig` sets `FAST_MODEL="mock"`/`DEEP_MODEL="mock"` which `reason_sot.llm.models.override_models` skips, so tests stay offline.

## Architecture

**ReasonSoT** is a dual-process (System 1 / System 2) reasoning engine for voice-interview agents. The core insight: a zero-cost heuristic router picks between a fast Haiku path and a deep Sonnet path with adaptive reasoning strategies, so most turns avoid System 2 latency entirely.

### Turn flow (the load-bearing orchestration)

`InterviewEngine.process_turn` in `reason_sot/interview/engine.py` is the entry point. Every turn walks these steps in order — they are NOT independent modules, they compose into one pipeline:

1. **KG update** (`interview/knowledge_graph.py`) — heuristic regex extraction of topics/entities/skills from user input, tied to persona-derived clusters. The KG summary is injected into a cached system block.
2. **Route** (`core/router.py`) — pure-Python scoring on regex patterns + technical-term density + phase/chain-depth signals. Output: `RoutingDecision(system, reasoning_mode, thinking_budget, beam_width)`. **No LLM call.**
3. **DST upgrade** — if router picked S2+CoT and KG coverage gap > 0.4, upgrade to DST with `estimate_beam_from_context`.
4. **Thinking budget** (`core/early_exit.py`) — scales `thinking_budget` from complexity + prior thinking analysis.
5. **Cache-optimized messages** (`llm/cache.py`) — up to 4 ephemeral cache breakpoints: base system, persona, KG summary, last user message. Breakpoint placement matters — changing the ordering busts cache.
6. **Stream via selected generator** (`_select_generator`):
   - `reasoning_mode == SPECULATIVE` → `core/speculative.py` (S1 draft + confidence marker → gate → optional S2 verify)
   - `system == 2 && mode == DST` → inject `build_dst_prompt(coverage_gaps, current_topic)` into the last user message, then `core/system2.generate`
   - `system == 2` (CoT/MoT) → `core/system2.generate` which prepends mode-specific instruction to the user message (NOT the system prompt — that would bust the prefix cache)
   - else → `core/system1.generate`
7. **Follow-up classify** (`interview/followup.py`) — S1 call returning `FollowUpAction` ∈ {NEXT_TOPIC, CLARIFY, EXPLORE}. Drives `_followup_chain_depth`, which feeds back into step 2 on the next turn.
8. **Persona switch suggestion** (`persona/manager.suggest_persona_switch`) — based on KG coverage + domain drift.
9. Record `InterviewTurn` with full usage/latency metrics.

### System 1 / System 2 split

- **System 1** (`core/system1.py`) — Haiku, no thinking, 512 max tokens. Used for simple routes, follow-up classification, and speculative drafts.
- **System 2** (`core/system2.py`) — Sonnet with extended thinking (`thinking_budget` from router). Thinking deltas are consumed internally and NOT streamed to callers — only `TextDelta` reaches the user. Reasoning modes (CoT, MoT, DST) differ only in the prompt prefix injected into the last user message.
- **Speculative** (`core/speculative.py`) — for borderline complexity. Appends `DRAFT_CONFIDENCE_PROMPT` to the system prompt, parses `[CONFIDENCE: 0.X]` from the draft, ships draft directly if confidence ≥ 0.8, else sends draft + `VERIFY_PROMPT` to S2. Combined `TokenUsage` and `LatencyMetrics` are synthesized from both calls (TTFT reported from the draft — the user sees S1 latency).

### Reasoning modes (all single-prompt — no multi-call fan-out)

- **CoT** (`core/system2.py::COT_INSTRUCTION`) — standard step-by-step.
- **MoT** (`core/mot.py`) — Matrix of Thought (R rows × C columns). `MoTConfig.from_signals` picks broad/deep/balanced. All exploration happens in thinking tokens, not separate API calls.
- **DST** (`core/dst.py`) — Domain-Specialized Tree with adaptive beam. Model self-assesses confidence in Step 1, skips Step 2 if confident. `estimate_beam_from_context` pre-scales the thinking budget.

### Streaming contract

`LLMClient.stream_message` (in `reason_sot/llm/client.py`) is the single LLM entry point. It yields `StreamEvent = TextDelta | ThinkingDelta | StreamDone | StreamError` (defined in `types.py`). When `thinking_budget` is set, `temperature` is forced to 1.0 (Anthropic API requirement).

### Persona layer

Personas are YAML files in `reason_sot/persona/definitions/`. `PersonaProfile` (`persona/profiles.py`) carries `topic_coverage` (priority 1=MUST/2=SHOULD/3=OPTIONAL), `decision_patterns`, `probing_strategies`. `render_system_prompt` returns `[base_system, persona_section]` — `BASE_SYSTEM_PROMPT` stays constant (first cache block) and the persona block lives in its own cache block so persona switches invalidate only that segment. `InterviewEngine.switch_persona` rebuilds the KG with the new persona's clusters and re-extracts from conversation history.

### Configuration

Tunables live in `config.py` as class attributes (not a dict) — read them via `get_config()` which returns a subclass based on `REASON_SOT_ENV`. Model IDs live in `reason_sot/llm/models.py::MODELS` — override via `override_models()` rather than editing directly.

### Testing conventions

`tests/conftest.py::MockLLMClient` is a full drop-in for `LLMClient` (same `stream_message`/`complete_message` signatures, returns canned `StreamEvent`s with synthetic latency). Use it instead of mocking `anthropic` directly. Tests marked `@pytest.mark.integration` hit the real API — keep them out of the default run.
