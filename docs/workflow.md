# Turn Workflow

This document walks through what happens end-to-end when a user utterance arrives — step by step, with sequence diagrams. The orchestration entry point is `InterviewEngine.process_turn(user_input)` in `reason_sot/interview/engine.py`.

If you want to read the code alongside the diagrams, open `reason_sot/interview/engine.py:143` (the `process_turn` method).

## The 9 steps

```mermaid
flowchart TD
    Start([user_input arrives]) --> S1[1. Update KG<br/>from utterance]
    S1 --> S2[2. Route<br/>router.route]
    S2 --> S3{3. KG gap > 0.4<br/>AND S2/CoT?}
    S3 -- yes --> S3a[upgrade to DST<br/>beam = estimate_beam]
    S3 -- no --> S4
    S3a --> S4[4. Estimate<br/>thinking budget]
    S4 --> S5[5. Build cache-optimized<br/>messages]
    S5 --> S6{6. Select<br/>generator}
    S6 -- Speculative --> G1[generate_speculative<br/>S1 draft → gate → S2]
    S6 -- S2 + DST --> G2[inject DST prompt →<br/>system2.generate]
    S6 -- S2 + CoT/MoT --> G3[system2.generate]
    S6 -- S1 --> G4[system1.generate]
    G1 --> Stream
    G2 --> Stream
    G3 --> Stream
    G4 --> Stream[Stream TextDelta<br/>events to caller]
    Stream --> S7[7. Classify follow-up<br/>S1 JSON call]
    S7 --> S8[8. Suggest<br/>persona switch?]
    S8 --> S9[9. Record<br/>InterviewTurn]
    S9 --> End([done])
```

## Sequence diagram — full turn

```mermaid
sequenceDiagram
    autonumber
    participant Caller as demo.py / caller
    participant Engine as InterviewEngine
    participant KG as KnowledgeGraph
    participant Router as router.route
    participant EE as early_exit
    participant Cache as cache.build_*
    participant Gen as selected generator<br/>(S1 / S2 / Speculative)
    participant API as Anthropic API
    participant FU as followup.classify

    Caller->>Engine: process_turn(user_input)
    Engine->>Engine: turn_count += 1, update phase

    Note over Engine,KG: Step 1 — KG update (heuristic, ~0ms)
    Engine->>KG: extract_from_turn(user, last_agent, turn#)
    KG-->>Engine: (KG mutated in place)
    Engine->>Engine: _rebuild_system_blocks()

    Note over Engine,Router: Step 2 — Route (microseconds, no LLM)
    Engine->>Router: route(user_input, phase, chain_depth, ...)
    Router-->>Engine: RoutingDecision

    alt KG gap > 0.4 and mode == COT
        Note over Engine: Step 3 — upgrade to DST
        Engine->>Engine: estimate_beam_from_context()
        Engine->>Engine: routing.reasoning_mode = DST
    end

    alt system == 2
        Note over Engine,EE: Step 4 — size thinking budget
        Engine->>EE: estimate_thinking_budget(complexity, mode, prev_analysis)
        EE-->>Engine: ThinkingBudgetEstimate
    end

    Note over Engine,Cache: Step 5 — cache-optimized messages
    Engine->>Cache: build_messages_with_cache(conversation)
    Cache-->>Engine: messages with cache_control markers

    Note over Engine,Gen: Step 6 — stream response
    Engine->>Gen: generate(client, messages, system_blocks, routing)
    Gen->>API: stream_message(model, thinking, cache blocks)
    loop stream events
        API-->>Gen: TextDelta / ThinkingDelta
        Gen-->>Engine: TextDelta (thinking consumed internally)
        Engine-->>Caller: TextDelta (real-time streaming)
    end
    API-->>Gen: final message + usage
    Gen-->>Engine: StreamDone(usage, latency)
    Engine-->>Caller: StreamDone

    Note over Engine,FU: Step 7 — follow-up classification
    Engine->>FU: classify_followup(agent_response, user_input, chain_depth)
    FU->>API: S1 call, JSON output
    API-->>FU: {"action": "clarify", ...}
    FU-->>Engine: FollowUpDecision
    Engine->>Engine: update chain_depth

    Note over Engine: Step 8 — persona switch check
    Engine->>Engine: suggest_persona_switch(coverage, topics)

    Note over Engine: Step 9 — record turn
    Engine->>Engine: session.turns.append(InterviewTurn(...))
    Engine-->>Caller: (iterator complete)
```

## The three generator paths

Step 6 selects one of four generators based on `RoutingDecision`:

```mermaid
flowchart TD
    R[RoutingDecision] --> D1{mode ==<br/>SPECULATIVE?}
    D1 -- yes --> SP[generate_speculative<br/>core/speculative.py]
    D1 -- no --> D2{system == 2?}
    D2 -- no --> S1G[system1.generate]
    D2 -- yes --> D3{mode == DST<br/>and KG present?}
    D3 -- yes --> DSTG[build_dst_prompt<br/>inject into last user msg<br/>→ system2.generate]
    D3 -- no --> S2G[system2.generate<br/>mode-specific instruction<br/>prepended to user msg]
```

### System 1 path

```mermaid
sequenceDiagram
    participant E as InterviewEngine
    participant S1 as system1.generate
    participant C as LLMClient
    participant A as Anthropic

    E->>S1: generate(client, messages, system_blocks)
    S1->>C: stream_message(model_tier=FAST, thinking_budget=None)
    C->>A: POST /messages (Haiku, stream=True, cache_control)
    loop
        A-->>C: content_block_delta
        C-->>S1: TextDelta
        S1-->>E: TextDelta (pass-through)
    end
    A-->>C: final_message
    C-->>S1: StreamDone(usage, latency)
    S1-->>E: StreamDone
```

Fast path. No thinking. Typical TTFT: 200–400 ms.

### System 2 path (CoT / MoT / DST)

```mermaid
sequenceDiagram
    participant E as InterviewEngine
    participant S2 as system2.generate
    participant C as LLMClient
    participant A as Anthropic

    E->>S2: generate(client, messages, system_blocks, routing)
    S2->>S2: _augment_messages_for_mode(messages, routing)
    Note right of S2: prepends mode instruction<br/>to last user message<br/>(NOT to system — cache!)
    S2->>C: stream_message(model_tier=DEEP, thinking_budget=N)
    C->>A: POST /messages (Sonnet, thinking=enabled, temp=1.0)
    loop
        A-->>C: thinking_delta (internal)
        C-->>S2: ThinkingDelta
        S2->>S2: consume, count words
        A-->>C: text_delta
        C-->>S2: TextDelta
        S2-->>E: TextDelta
    end
    A-->>C: final_message
    C-->>S2: StreamDone(usage incl. thinking_tokens)
    S2-->>E: StreamDone
```

Thinking deltas are consumed internally — never streamed to the caller. The thinking text is captured and passed to `analyze_thinking()` to adjust the next turn's budget multiplier.

### Speculative path

```mermaid
sequenceDiagram
    participant E as InterviewEngine
    participant Sp as generate_speculative
    participant C as LLMClient
    participant A as Anthropic

    Note over Sp: Phase 1 — DRAFT
    Sp->>Sp: _augment_system_for_confidence(system)
    Sp->>C: complete_message(model=FAST, +confidence prompt)
    C->>A: Haiku call
    A-->>C: "Draft reply.\n[CONFIDENCE: 0.X]"
    C-->>Sp: (draft_text, usage, latency)
    Sp->>Sp: _extract_confidence(draft_text)

    alt confidence >= 0.8
        Note over Sp: Phase 2 — SHIP DRAFT
        Sp-->>E: TextDelta(clean_draft)
        Sp-->>E: StreamDone(draft_usage, draft_latency)
    else confidence < 0.8
        Note over Sp: Phase 3 — VERIFY with S2
        Sp->>C: stream_message(model=DEEP, messages+VERIFY_PROMPT)
        C->>A: Sonnet call with extended thinking
        loop
            A-->>C: deltas
            C-->>Sp: TextDelta
            Sp-->>E: TextDelta
        end
        A-->>C: final
        C-->>Sp: StreamDone
        Sp->>Sp: combine usage (draft + verify)
        Sp->>Sp: latency = (draft.start, draft.ttft, verify.end)
        Sp-->>E: StreamDone(combined)
    end
```

The combined `LatencyMetrics` uses the draft's TTFT (what the user *perceives*) and the verify call's end-time (the true end of generation). This is how 48–66% effective latency reductions are possible on borderline turns.

## State updates per turn

After step 9 completes, these have changed:

| State | Updated by | New value depends on |
|---|---|---|
| `_turn_count` | `process_turn` top | `+1` |
| `_phase` | `_update_phase()` | `turn_count`, `followup_chain_depth`, KG coverage gap |
| `_conversation` | steps 1 + 6 | appended user + assistant messages |
| `_kg` nodes/edges/coverage | step 1 | extracted from `user_input` + last agent reply |
| `_system_blocks` | step 1 (after KG) | `[base_system, persona_prompt, kg_summary]` |
| `_last_thinking_analysis` | step 6 (S2 only) | `analyze_thinking(thinking_text)` |
| `_followup_chain_depth` | step 7 | `+1` on CLARIFY, `0` on NEXT_TOPIC or EXPLORE |
| `_current_topic` | step 7 | from `FollowUpDecision.topic` |
| `_persona_switch_suggested` | step 8 | persona name or `None` |
| `_session.turns` | step 9 | `.append(InterviewTurn(...))` |

## Phase transitions

`_update_phase()` runs at the top of every turn:

```mermaid
stateDiagram-v2
    [*] --> OPENING: turn 1
    OPENING --> CORE: turn > 2
    CORE --> DEEP_DIVE: chain_depth >= 2
    CORE --> DEEP_DIVE: kg_gap_score < 0.3
    DEEP_DIVE --> CORE: chain_depth resets
    CORE --> CLOSING: turn >= 25
    DEEP_DIVE --> CLOSING: turn >= 25
    CLOSING --> [*]
```

Phase is a **routing signal** — it feeds into `router._compute_complexity_score()`:

| Phase | Complexity bias |
|---|---|
| `OPENING` | −0.2 (push toward S1 — rapport building) |
| `CORE` | 0.0 (no bias) |
| `DEEP_DIVE` | +0.2 (push toward S2 — thorough probing) |
| `CLOSING` | −0.15 (back to S1 — wrap up) |

## Streaming contract

`process_turn` is an `AsyncIterator[StreamEvent]`. Callers consume it like this (from `demo.py`):

```python
async for event in engine.process_turn(user_input):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, StreamDone):
        # per-turn metrics available in event.usage and event.latency
        ...
    elif isinstance(event, StreamError):
        # LLM call failed — decide whether to retry or bail
        ...
```

Three event types reach the caller:

- **`TextDelta`** — a chunk of the visible response. Stream straight to TTS / stdout.
- **`StreamDone`** — emitted once at the end of each turn. Contains `usage` (tokens + cache hit rate) and `latency` (TTFT, total_ms).
- **`StreamError`** — the LLM call errored. The engine has already logged the problem; handle as you see fit.

`ThinkingDelta` events are **not** yielded to the caller. They're consumed inside `system2.generate` and `speculative.generate_speculative`.

## Failure modes

| What fails | What happens |
|---|---|
| Anthropic API error mid-stream | `StreamError` yielded; partial response (if any) still recorded |
| Follow-up classification fails | Heuristic fallback in `followup._heuristic_classify()` |
| Speculative draft missing confidence marker | Defaults to `confidence=0.5` → triggers verify path |
| Speculative S2 verify fails | Falls back to shipping the draft as-is |
| Persona file missing at `switch_persona` | Returns `False`, keeps current persona |
| KG extraction throws | Logged; turn continues with stale KG summary |

The engine is built to **always produce a turn record** even on failure, so session analytics stay consistent.

## Next

- [Routing design](./design/routing.md) — how step 2 actually works.
- [Reasoning modes](./design/reasoning-modes.md) — what happens inside steps 5–6 for each mode.
- [Prefix caching](./design/prefix-caching.md) — why step 5's cache placement matters.
