"""Class-based configuration for ReasonSoT."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""

    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    FAST_MODEL: str = "claude-haiku-4-5-20251001"
    DEEP_MODEL: str = "claude-sonnet-4-6-20250514"

    # Routing thresholds
    COMPLEXITY_THRESHOLD: float = 0.6
    CONFIDENCE_THRESHOLD: float = 0.8

    # Latency budgets (ms)
    S1_TIMEOUT_MS: int = 5000
    S2_TIMEOUT_MS: int = 15000
    S1_MAX_TOKENS: int = 512
    S2_MAX_TOKENS: int = 2048
    S2_DEFAULT_THINKING_BUDGET: int = 4096
    S2_MAX_THINKING_BUDGET: int = 10000

    # Domain-Specialized Tree
    DST_MIN_BEAM_WIDTH: int = 1
    DST_MAX_BEAM_WIDTH: int = 3
    DST_CONFIDENCE_EXPAND_THRESHOLD: float = 0.5

    # Matrix of Thought
    MOT_DEFAULT_ROWS: int = 3
    MOT_DEFAULT_COLS: int = 2

    # Interview
    MAX_TURNS: int = 30
    MAX_FOLLOWUP_CHAIN: int = 3

    # Logging
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    DEBUG: bool = False


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    LOG_LEVEL = "INFO"


class TestConfig(Config):
    ANTHROPIC_API_KEY = "test-key"
    FAST_MODEL = "mock"
    DEEP_MODEL = "mock"
    DEBUG = True
    LOG_LEVEL = "DEBUG"


_env_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestConfig,
}


def get_config() -> Config:
    env = os.environ.get("REASON_SOT_ENV", "development")
    return _env_map.get(env, DevelopmentConfig)()
