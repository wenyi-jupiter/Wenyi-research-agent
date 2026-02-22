"""Configuration management for the agent engine."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM API Keys - Minimax
    minimax_api_key: str = Field(default="", description="Minimax API key")
    minimax_group_id: str = Field(default="", description="Minimax Group ID")

    # LLM API Keys - Qwen (Aliyun DashScope)
    dashscope_api_key: str = Field(default="", description="Aliyun DashScope API key for Qwen")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope API base URL"
    )

    # Default LLM Provider
    default_llm_provider: Literal["minimax", "qwen"] = Field(
        default="qwen", description="Default LLM provider"
    )
    default_minimax_model: str = Field(
        default="abab6.5-chat", description="Default Minimax model"
    )
    default_qwen_model: str = Field(
        default="qwen-plus", description="Default Qwen model"
    )

    # Qwen Web Search Configuration (DashScope built-in)
    qwen_enable_search: bool = Field(
        default=True, description="Enable web search for Qwen models"
    )
    qwen_search_strategy: Literal["turbo", "max", "agent"] = Field(
        default="turbo", description="Search strategy: turbo (fast), max (thorough), agent (multi-round)"
    )
    qwen_forced_search: bool = Field(
        default=False, description="Force web search even if model thinks it's not needed"
    )

    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://agent_user:agent_password@localhost:5432/agent_engine",
        description="PostgreSQL connection URL",
    )

    # Embedding Configuration
    embedding_provider: Literal["mock", "dashscope"] = Field(
        default="dashscope", description="Embedding provider (mock or dashscope)"
    )
    embedding_model: str = Field(
        default="text-embedding-v4", description="Embedding model name"
    )
    embedding_dimension: int = Field(
        default=1024, description="Embedding vector dimension (text-embedding-v4 supports: 2048, 1536, 1024, 768, 512, 256, 128, 64)"
    )

    # Execution Limits
    # NOTE: Multi-turn executor consumes significantly more tokens than single-turn.
    # Each subtask may use 5-10 LLM calls with growing context windows.
    # A 5-subtask plan with multi-turn can easily consume 300k-500k tokens.
    max_tokens: int = Field(default=500_000, description="Maximum tokens per task")
    max_steps: int = Field(default=100, description="Maximum execution steps (LLM calls)")
    max_tool_calls: int = Field(default=200, description="Maximum tool invocations")
    execution_timeout: int = Field(default=1200, description="Execution timeout in seconds")

    # Retry Configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts for tool calls")
    retry_base_delay: float = Field(default=1.0, description="Base delay for exponential backoff")
    tool_timeout: int = Field(default=180, description="Timeout per tool call in seconds")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")

    # ── Per-Agent Model Configuration ──
    # Each agent role can use a different model to balance cost vs quality.
    # All models are accessed via DashScope API (same endpoint, different model names).
    # Cost tiers (approximate):
    #   qwen-turbo   — cheapest, fastest (~0.3¥/M tokens)
    #   qwen-plus    — mid-tier, good balance (~2¥/M tokens)
    #   MiniMax-M2.1 — high quality, expensive (~10¥/M tokens)
    #
    # Strategy: use MiniMax-M2.1 for quality-critical roles (planner, reporter, extractor),
    # and qwen-plus for high-volume roles (executor tool loops, critic).
    planner_model: str = Field(
        default="MiniMax-M2.1",
        description=(
            "Model for the planner agent (task decomposition). "
            "MiniMax-M2.1 produces higher quality subtask plans with better structure."
        ),
    )
    executor_model: str = Field(
        default="qwen-plus",
        description=(
            "Model for the executor agent (tool calling loop). "
            "This is the highest-volume caller (~80% of tokens). "
            "qwen-plus is a good balance; qwen-turbo saves cost but may miss nuances."
        ),
    )
    critic_model: str = Field(
        default="qwen-plus",
        description=(
            "Model for the critic agent (quality evaluation). "
            "qwen-plus is sufficient since we now also have programmatic scoring."
        ),
    )
    reporter_model: str = Field(
        default="MiniMax-M2.1",
        description=(
            "Model for the reporter agent (final report generation). "
            "Use the best available model here — report quality is the user-facing output. "
            "MiniMax-M2.1 produces more fluent, well-structured reports."
        ),
    )
    extractor_model: str = Field(
        default="MiniMax-M2.1",
        description=(
            "Model for the data extractor role (precise data extraction from content). "
            "High-quality model recommended for accurate number/fact extraction."
        ),
    )

    # Skills Directory
    skills_dir: str = Field(default="./skills", description="Directory for skill files")

    # HTTP fetching configuration
    http_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        description=(
            "Default User-Agent for HTTP fetching tools. "
            "When curl_cffi is installed, the SmartHttpClient also rotates "
            "through a pool of realistic browser UAs automatically."
        ),
    )
    sec_user_agent: str = Field(
        default="agent_demo/0.1 (SEC EDGAR; contact: set SEC_USER_AGENT env var)",
        description=(
            "User-Agent for SEC endpoints. SEC requests may be blocked without a "
            "descriptive UA string. Override via SEC_USER_AGENT env var."
        ),
    )
    http_max_retries: int = Field(
        default=3,
        description=(
            "Maximum retry attempts for HTTP requests on 403/429 errors. "
            "Each retry rotates the TLS fingerprint and User-Agent."
        ),
    )
    playwright_enabled: bool = Field(
        default=True,
        description=(
            "Enable Playwright headless browser fallback for JS-rendered pages. "
            "Requires 'playwright' package and 'playwright install chromium'."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
