"""Interview session manager — the top-level orchestrator.

Phase 3: Full engine with KG, DST, speculative CoT, and persona switching.

Per turn:
  1. Receive user input
  2. Update knowledge graph with new information
  3. Route via heuristic classifier (informed by KG coverage gaps)
  4. Estimate thinking budget (early_exit)
  5. For DST: inject coverage gap context into reasoning prompt
  6. Stream response through S1 / S2 / Speculative path
  7. Classify follow-up action
  8. Check for persona switch suggestion
  9. Record turn with full metrics
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from reason_sot.core import router, system1, system2
from reason_sot.core.dst import DSTConfig, build_dst_prompt, estimate_beam_from_context
from reason_sot.core.early_exit import (
    ThinkingAnalysis,
    analyze_thinking,
    estimate_thinking_budget,
)
from reason_sot.core.speculative import generate_speculative
from reason_sot.interview.followup import classify_followup
from reason_sot.interview.knowledge_graph import KnowledgeGraph
from reason_sot.llm.cache import build_messages_with_cache, build_system_blocks
from reason_sot.llm.client import LLMClient
from reason_sot.persona.manager import (
    load_persona,
    render_system_prompt,
    suggest_persona_switch,
)
from reason_sot.persona.profiles import PersonaProfile
from reason_sot.types import (
    FollowUpAction,
    FollowUpDecision,
    InterviewPhase,
    InterviewSession,
    InterviewTurn,
    LatencyMetrics,
    ReasoningMode,
    RoutingDecision,
    StreamDone,
    StreamError,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class InterviewEngine:
    """Manages an interview session from start to finish.

    Full orchestration of the ReasonSoT reasoning engine:
    - Knowledge graph for structured interview context
    - Heuristic router informed by KG coverage gaps
    - System 1 (Haiku) for fast responses
    - System 2 (Sonnet) with MoT/DST reasoning strategies
    - Speculative CoT for borderline cases
    - Adaptive follow-up classification
    - Early exit / thinking budget optimization
    - Dynamic persona switching
    """

    def __init__(
        self,
        client: LLMClient,
        persona: PersonaProfile,
        complexity_threshold: float = 0.6,
        max_followup_chain: int = 3,
        enable_kg: bool = True,
        enable_speculative: bool = True,
    ) -> None:
        self._client = client
        self._persona = persona
        self._complexity_threshold = complexity_threshold
        self._max_followup_chain = max_followup_chain
        self._enable_kg = enable_kg
        self._enable_speculative = enable_speculative

        self._conversation: list[dict[str, Any]] = []
        self._session = InterviewSession(
            session_id=uuid.uuid4().hex[:12],
            persona_name=persona.name,
        )
        self._phase = InterviewPhase.OPENING
        self._turn_count = 0
        self._followup_chain_depth = 0
        self._current_topic = ""
        self._last_follow_up: FollowUpDecision | None = None
        self._last_thinking_analysis: ThinkingAnalysis | None = None
        self._last_agent_confidence = 1.0
        self._persona_switch_suggested: str | None = None

        # Knowledge graph
        self._kg = KnowledgeGraph(persona) if enable_kg else None

        # Pre-render system prompt blocks (cached across turns)
        self._rebuild_system_blocks()

    def _rebuild_system_blocks(self) -> None:
        """Rebuild system prompt blocks (call after persona switch or KG update)."""
        base_system, persona_prompt = render_system_prompt(self._persona)
        kg_summary = self._kg.to_summary() if self._kg and self._turn_count > 0 else None
        self._system_blocks = build_system_blocks(
            base_system=base_system,
            persona_prompt=persona_prompt,
            kg_summary=kg_summary,
        )

    @property
    def session(self) -> InterviewSession:
        return self._session

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def current_phase(self) -> InterviewPhase:
        return self._phase

    @property
    def knowledge_graph(self) -> KnowledgeGraph | None:
        return self._kg

    def get_opening_message(self) -> str:
        """Return the persona's opening message."""
        return self._persona.opening_message.strip()

    async def process_turn(self, user_input: str) -> AsyncIterator[StreamEvent]:
        """Process a single interview turn with streaming.

        Yields StreamEvent objects for real-time output.
        """
        self._turn_count += 1
        self._update_phase()

        # Add user input to conversation
        self._conversation.append({"role": "user", "content": user_input})

        # ── Step 1: Update knowledge graph ─────────────────────────────
        if self._kg:
            last_agent = ""
            if len(self._conversation) >= 2:
                prev = self._conversation[-2]
                if prev.get("role") == "assistant":
                    last_agent = prev.get("content", "")
            self._kg.extract_from_turn(user_input, last_agent, self._turn_count)
            # Rebuild system blocks with updated KG summary
            self._rebuild_system_blocks()

        # ── Step 2: Route (informed by KG) ─────────────────────────────
        routing = router.route(
            user_input=user_input,
            turn_number=self._turn_count,
            phase=self._phase,
            followup_chain_depth=self._followup_chain_depth,
            previous_confidence=self._last_agent_confidence,
            complexity_threshold=self._complexity_threshold,
        )

        # Enrich routing with KG coverage gaps for DST
        if self._kg and routing.system == 2 and routing.reasoning_mode == ReasoningMode.COT:
            # Consider upgrading to DST if coverage gaps exist
            gap_score = self._kg.get_coverage_gap_score()
            if gap_score > 0.4:
                beam = estimate_beam_from_context(
                    previous_confidence=self._last_agent_confidence,
                    followup_chain_depth=self._followup_chain_depth,
                    coverage_gap_score=gap_score,
                )
                if beam > 1:
                    routing.reasoning_mode = ReasoningMode.DST
                    routing.beam_width = beam
                    routing.rationale += f" (upgraded to DST, beam={beam}, gap={gap_score:.2f})"

        # ── Step 3: Estimate thinking budget ───────────────────────────
        if routing.system == 2:
            budget_est = estimate_thinking_budget(
                complexity_score=1.0 - routing.confidence,
                reasoning_mode=routing.reasoning_mode.value,
                previous_analysis=self._last_thinking_analysis,
            )
            routing.thinking_budget = budget_est.budget_tokens
            logger.info(
                "Turn %d: S2 routing — mode=%s budget=%d beam=%d (%s)",
                self._turn_count,
                routing.reasoning_mode.value,
                routing.thinking_budget,
                routing.beam_width,
                routing.rationale,
            )
        else:
            logger.info(
                "Turn %d: S1 routing — mode=%s (%s)",
                self._turn_count,
                routing.reasoning_mode.value,
                routing.rationale,
            )

        # ── Step 4: Build cache-optimized messages ��────────────────────
        cached_messages = build_messages_with_cache(self._conversation)

        # ── Step 5: Stream response ────────────────────────────────────
        response_chunks: list[str] = []
        thinking_chunks: list[str] = []
        usage = TokenUsage()
        latency = LatencyMetrics(start_time=time.monotonic())

        event_stream = self._select_generator(routing, cached_messages)

        async for event in event_stream:
            if isinstance(event, TextDelta):
                response_chunks.append(event.text)
                yield event
            elif isinstance(event, ThinkingDelta):
                thinking_chunks.append(event.text)
            elif isinstance(event, StreamDone):
                usage = event.usage
                latency = event.latency
                yield event
            elif isinstance(event, StreamError):
                yield event
                return

        full_response = "".join(response_chunks)
        self._conversation.append({"role": "assistant", "content": full_response})

        # ── Step 6: Analyze thinking ───────────────────────────────────
        if thinking_chunks:
            thinking_text = "".join(thinking_chunks)
            self._last_thinking_analysis = analyze_thinking(thinking_text)

        # ── Step 7: Classify follow-up ─────────────────────────────────
        follow_up = await self._classify_followup(full_response, user_input)
        self._update_followup_state(follow_up)

        # ── Step 8: Check persona switch ───────────────────────────────
        if self._kg:
            self._persona_switch_suggested = suggest_persona_switch(
                current_persona=self._persona,
                conversation_signals={
                    "topic_coverage": self._kg.get_coverage(),
                    "dominant_topics": [],
                },
            )
            if self._persona_switch_suggested:
                logger.info(
                    "Persona switch suggested: %s → %s",
                    self._persona.name,
                    self._persona_switch_suggested,
                )

        # ── Step 9: Record turn ────────────────────────────────────────
        model_name = "sonnet" if routing.system == 2 else "haiku"
        if routing.reasoning_mode == ReasoningMode.SPECULATIVE:
            model_name = "haiku+sonnet"
        turn = InterviewTurn(
            turn_number=self._turn_count,
            user_input=user_input,
            agent_response=full_response,
            routing=routing,
            follow_up=follow_up,
            usage=usage,
            latency=latency,
            phase=self._phase,
            model_used=model_name,
        )
        self._session.turns.append(turn)

        logger.info(
            "Turn %d [S%d/%s]: TTFT=%.0fms total=%.0fms tokens=%d cache=%.0f%% → %s",
            self._turn_count,
            routing.system,
            routing.reasoning_mode.value,
            latency.ttft_ms or 0,
            latency.total_ms or 0,
            usage.output_tokens,
            usage.cache_hit_rate * 100,
            follow_up.action.value if follow_up else "none",
        )

    def _select_generator(
        self,
        routing: RoutingDecision,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        """Select the appropriate response generator based on routing."""
        # Speculative path: draft with S1, verify with S2 if needed
        if routing.reasoning_mode == ReasoningMode.SPECULATIVE and self._enable_speculative:
            return generate_speculative(
                client=self._client,
                messages=messages,
                system=self._system_blocks,
                confidence_threshold=0.8,
                s2_thinking_budget=routing.thinking_budget or 2048,
            )

        # System 2: deep reasoning with MoT/DST/CoT
        if routing.system == 2:
            # For DST mode, augment with coverage gap context
            if routing.reasoning_mode == ReasoningMode.DST and self._kg:
                gaps = self._kg.get_uncovered_topics()
                dst_prompt = build_dst_prompt(
                    coverage_gaps=gaps,
                    current_topic=self._current_topic,
                )
                # Inject DST prompt into the last user message
                augmented = [m.copy() for m in messages]
                for i in range(len(augmented) - 1, -1, -1):
                    if augmented[i].get("role") == "user":
                        content = augmented[i].get("content", "")
                        if isinstance(content, str):
                            augmented[i]["content"] = f"{dst_prompt}\n\n{content}"
                        break
                return system2.generate(
                    client=self._client,
                    messages=augmented,
                    system=self._system_blocks,
                    routing=routing,
                )

            return system2.generate(
                client=self._client,
                messages=messages,
                system=self._system_blocks,
                routing=routing,
            )

        # System 1: fast path
        return system1.generate(
            client=self._client,
            messages=messages,
            system=self._system_blocks,
        )

    async def switch_persona(self, persona_name: str) -> bool:
        """Switch to a different persona mid-interview.

        Returns True if switch was successful.
        """
        try:
            new_persona = load_persona(persona_name)
        except FileNotFoundError:
            logger.error("Cannot switch to persona '%s': not found", persona_name)
            return False

        old_name = self._persona.name
        self._persona = new_persona
        self._session.persona_name = new_persona.name

        # Rebuild KG with new persona's topics
        if self._enable_kg:
            self._kg = KnowledgeGraph(new_persona)
            # Re-extract from existing conversation
            for i in range(0, len(self._conversation) - 1, 2):
                user_msg = self._conversation[i].get("content", "")
                agent_msg = self._conversation[i + 1].get("content", "") if i + 1 < len(self._conversation) else ""
                self._kg.extract_from_turn(user_msg, agent_msg, i // 2 + 1)

        # Rebuild system blocks with new persona
        self._rebuild_system_blocks()
        self._persona_switch_suggested = None

        logger.info("Persona switched: %s → %s", old_name, new_persona.name)
        return True

    async def _classify_followup(
        self, agent_response: str, user_input: str
    ) -> FollowUpDecision:
        """Classify the follow-up action for this turn."""
        if self._phase in (InterviewPhase.OPENING, InterviewPhase.CLOSING):
            return FollowUpDecision(
                action=FollowUpAction.NEXT_TOPIC,
                reason=f"Phase: {self._phase.value}",
            )

        try:
            return await classify_followup(
                client=self._client,
                agent_question=agent_response,
                candidate_response=user_input,
                followup_chain_depth=self._followup_chain_depth,
                max_chain=self._max_followup_chain,
            )
        except Exception as e:
            logger.warning("Follow-up classification failed: %s", e)
            return FollowUpDecision(
                action=FollowUpAction.CLARIFY,
                reason=f"Classification error: {e}",
            )

    def _update_followup_state(self, follow_up: FollowUpDecision) -> None:
        """Update follow-up chain tracking."""
        if follow_up.action == FollowUpAction.NEXT_TOPIC:
            self._followup_chain_depth = 0
            if follow_up.topic:
                self._current_topic = follow_up.topic
        elif follow_up.action == FollowUpAction.CLARIFY:
            self._followup_chain_depth += 1
        elif follow_up.action == FollowUpAction.EXPLORE:
            self._followup_chain_depth = 0
            if follow_up.topic:
                self._current_topic = follow_up.topic

        self._last_follow_up = follow_up

    def _update_phase(self) -> None:
        """Update interview phase based on turn count and KG coverage."""
        if self._turn_count <= 2:
            self._phase = InterviewPhase.OPENING
        elif self._turn_count >= 25:
            self._phase = InterviewPhase.CLOSING
        elif self._followup_chain_depth >= 2:
            self._phase = InterviewPhase.DEEP_DIVE
        elif self._kg and self._kg.get_coverage_gap_score() < 0.3:
            # Most topics covered — shift to deep dive on remaining gaps
            self._phase = InterviewPhase.DEEP_DIVE
        else:
            self._phase = InterviewPhase.CORE

    async def end_session(self) -> InterviewSession:
        """End the interview and return the complete session."""
        self._session.ended_at = time.time()
        if self._kg:
            self._session.knowledge_graph_snapshot = self._kg.to_snapshot()
        return self._session

    def get_session_stats(self) -> dict[str, Any]:
        """Get current session statistics."""
        s1_turns = [t for t in self._session.turns if t.routing.system == 1]
        s2_turns = [t for t in self._session.turns if t.routing.system == 2]

        s1_ttfts = [t.latency.ttft_ms for t in s1_turns if t.latency.ttft_ms]
        s2_ttfts = [t.latency.ttft_ms for t in s2_turns if t.latency.ttft_ms]

        stats: dict[str, Any] = {
            "total_turns": self._turn_count,
            "s1_turns": len(s1_turns),
            "s2_turns": len(s2_turns),
            "s1_avg_ttft_ms": sum(s1_ttfts) / len(s1_ttfts) if s1_ttfts else None,
            "s2_avg_ttft_ms": sum(s2_ttfts) / len(s2_ttfts) if s2_ttfts else None,
            "current_phase": self._phase.value,
            "followup_chain_depth": self._followup_chain_depth,
            "reasoning_modes_used": list(set(
                t.routing.reasoning_mode.value for t in self._session.turns
            )),
        }

        if self._kg:
            stats["kg_coverage"] = self._kg.get_coverage()
            stats["kg_gaps"] = self._kg.get_uncovered_topics()
            stats["kg_suggested_next"] = self._kg.get_suggested_next_topic()
            stats["kg_nodes"] = len(self._kg._nodes)
            stats["kg_edges"] = len(self._kg._edges)

        if self._persona_switch_suggested:
            stats["persona_switch_suggested"] = self._persona_switch_suggested

        return stats
