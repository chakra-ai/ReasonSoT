"""Model registry mapping tiers to Anthropic model IDs."""

from __future__ import annotations

from dataclasses import dataclass

from reason_sot.types import ModelTier


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model."""

    model_id: str
    tier: ModelTier
    max_output_tokens: int
    supports_thinking: bool
    timeout_ms: int


# Default model registry
MODELS: dict[ModelTier, ModelSpec] = {
    ModelTier.FAST: ModelSpec(
        model_id="claude-haiku-4-5-20251001",
        tier=ModelTier.FAST,
        max_output_tokens=512,
        supports_thinking=True,
        timeout_ms=5000,
    ),
    ModelTier.DEEP: ModelSpec(
        model_id="claude-sonnet-4-6-20250514",
        tier=ModelTier.DEEP,
        max_output_tokens=2048,
        supports_thinking=True,
        timeout_ms=15000,
    ),
}


def get_model(tier: ModelTier) -> ModelSpec:
    return MODELS[tier]


def override_models(fast_id: str | None = None, deep_id: str | None = None) -> None:
    """Override model IDs from config (e.g., for testing)."""
    if fast_id and fast_id != "mock":
        MODELS[ModelTier.FAST] = ModelSpec(
            model_id=fast_id,
            tier=ModelTier.FAST,
            max_output_tokens=MODELS[ModelTier.FAST].max_output_tokens,
            supports_thinking=MODELS[ModelTier.FAST].supports_thinking,
            timeout_ms=MODELS[ModelTier.FAST].timeout_ms,
        )
    if deep_id and deep_id != "mock":
        MODELS[ModelTier.DEEP] = ModelSpec(
            model_id=deep_id,
            tier=ModelTier.DEEP,
            max_output_tokens=MODELS[ModelTier.DEEP].max_output_tokens,
            supports_thinking=MODELS[ModelTier.DEEP].supports_thinking,
            timeout_ms=MODELS[ModelTier.DEEP].timeout_ms,
        )
