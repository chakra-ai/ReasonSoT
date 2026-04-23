"""Matrix of Thought (MoT) — breadth + depth reasoning in a single prompt.

Based on the MoT paper (2509.03918): unifies Chain-of-Thought (depth-only)
and Tree-of-Thought (breadth-only) into a matrix structure where:
  - Rows (R) = number of parallel reasoning paths (breadth)
  - Columns (C) = depth steps per path (depth)

When R=1, C=many → degenerates to CoT (deep, single path)
When R=many, C=1 → degenerates to ToT-like (broad exploration)
When R>1, C>1 → full MoT (breadth AND depth)

Implementation: single-prompt approach that instructs the model to explore
an R x C matrix within its thinking budget. No multi-call overhead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MoTConfig:
    """Configuration for a Matrix of Thought exploration."""

    rows: int = 3  # Number of parallel paths (breadth)
    cols: int = 2  # Depth steps per path
    mode: str = "balanced"  # "broad", "deep", or "balanced"

    @classmethod
    def broad(cls) -> MoTConfig:
        """Broad exploration: many paths, shallow depth."""
        return cls(rows=4, cols=1, mode="broad")

    @classmethod
    def deep(cls) -> MoTConfig:
        """Deep probing: single path, many steps."""
        return cls(rows=1, cols=4, mode="deep")

    @classmethod
    def balanced(cls) -> MoTConfig:
        """Balanced: moderate breadth and depth."""
        return cls(rows=3, cols=2, mode="balanced")

    @classmethod
    def from_signals(
        cls,
        deep_pattern_count: int = 0,
        topic_coverage_gap: float = 0.0,
    ) -> MoTConfig:
        """Auto-select MoT config from routing signals.

        Args:
            deep_pattern_count: How many deep-thinking patterns were detected.
            topic_coverage_gap: Fraction of topics not yet explored (0-1).
        """
        # Many uncovered topics → go broad to explore
        if topic_coverage_gap > 0.5:
            return cls.broad()

        # Multiple deep-thinking signals → go deep
        if deep_pattern_count >= 2:
            return cls.deep()

        return cls.balanced()


def build_mot_prompt(
    context: str = "the current situation",
    config: MoTConfig | None = None,
) -> str:
    """Build a Matrix of Thought instruction prompt.

    This is injected into the user message to guide the model's extended
    thinking. The model explores an R x C matrix of reasoning paths within
    its thinking budget, then synthesizes the best path into a response.
    """
    cfg = config or MoTConfig.balanced()

    if cfg.mode == "deep":
        return _build_deep_prompt(cfg, context)
    elif cfg.mode == "broad":
        return _build_broad_prompt(cfg, context)
    else:
        return _build_balanced_prompt(cfg, context)


def _build_balanced_prompt(cfg: MoTConfig, context: str) -> str:
    return f"""[REASONING STRATEGY: Matrix of Thought — {cfg.rows} paths x {cfg.cols} depth steps]

In your thinking, explore {context} using a structured matrix approach:

STEP 1 — BREADTH: Generate {cfg.rows} distinct angles or directions you could take.
For each angle, write a one-line description.

STEP 2 — DEPTH: For each of the {cfg.rows} angles, reason {cfg.cols} steps deep:
  - Step 1: What does this angle reveal about the candidate's understanding?
  - Step 2: What's the most incisive follow-up question from this angle?

STEP 3 — SYNTHESIS: Compare the {cfg.rows} paths. Select the single best follow-up
question that maximizes information gain about the candidate's true capability.

Your spoken response should be natural and conversational — the matrix reasoning
happens in your thinking only. Ask ONE question."""


def _build_deep_prompt(cfg: MoTConfig, context: str) -> str:
    return f"""[REASONING STRATEGY: Deep Chain — {cfg.cols} reasoning steps]

In your thinking, analyze {context} by going {cfg.cols} levels deep:

Level 1: What did the candidate actually demonstrate?
Level 2: What underlying mental model does this reveal?
Level 3: Where is the boundary of their understanding?
Level 4: What question would expose that boundary precisely?

Your spoken response should be natural — deep analysis in thinking only. Ask ONE question."""


def _build_broad_prompt(cfg: MoTConfig, context: str) -> str:
    return f"""[REASONING STRATEGY: Broad Exploration — {cfg.rows} directions]

In your thinking, consider {context} from {cfg.rows} different angles:

For each angle:
  - What topic area does this open up?
  - Why is this worth exploring right now?
  - What's a good opening question for this direction?

Then select the most promising direction based on: what we haven't covered yet,
what the candidate seems strongest/weakest in, and what will give us the most signal.

Your spoken response should be natural — broad reasoning in thinking only. Ask ONE question."""


@dataclass
class MoTResult:
    """Parsed result from a Matrix of Thought exploration."""

    selected_path: int = 0
    paths_explored: int = 0
    depth_reached: int = 0
    synthesis: str = ""


def parse_mot_response(thinking_text: str) -> MoTResult:
    """Parse the thinking output to extract MoT structure.

    This is used for scoring and analysis, not for runtime decisions.
    The model's visible response is already synthesized.
    """
    # Count how many distinct paths/angles were explored
    path_markers = re.findall(
        r"(?:angle|path|direction|option)\s*(\d+)", thinking_text, re.IGNORECASE
    )
    paths = len(set(path_markers)) if path_markers else 1

    # Count depth levels
    depth_markers = re.findall(
        r"(?:level|step|layer)\s*(\d+)", thinking_text, re.IGNORECASE
    )
    depth = max((int(d) for d in depth_markers), default=1)

    # Check for synthesis/selection
    has_synthesis = bool(
        re.search(r"(?:select|best|choose|synthesis|comparing)", thinking_text, re.IGNORECASE)
    )

    return MoTResult(
        selected_path=1,
        paths_explored=paths,
        depth_reached=depth,
        synthesis="present" if has_synthesis else "absent",
    )
