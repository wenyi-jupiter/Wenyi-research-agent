"""LLM provider router for dynamic provider selection."""

from typing import Literal

from agent_engine.config import get_settings
from agent_engine.llm.base import BaseLLMProvider
from agent_engine.llm.minimax import MinimaxProvider
from agent_engine.llm.qwen import QwenProvider


ProviderType = Literal["minimax", "qwen"]

_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "minimax": MinimaxProvider,
    "qwen": QwenProvider,
}


class LLMRouter:
    """Router for selecting and managing LLM providers."""

    def __init__(self):
        """Initialize the router with cached providers."""
        self._cache: dict[str, BaseLLMProvider] = {}

    def get_provider(
        self,
        provider: ProviderType | None = None,
        model: str | None = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Get an LLM provider instance.

        Args:
            provider: Provider type ("minimax", "qwen"). Defaults to settings.
            model: Optional model override.
            **kwargs: Additional provider arguments.

        Returns:
            An LLM provider instance.

        Raises:
            ValueError: If provider is unknown.
        """
        settings = get_settings()
        provider = provider or settings.default_llm_provider

        if provider not in _PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {list(_PROVIDERS.keys())}"
            )

        # Create cache key
        cache_key = f"{provider}:{model or 'default'}"

        # Return cached instance if no custom kwargs
        if not kwargs and cache_key in self._cache:
            return self._cache[cache_key]

        # Create new provider instance
        provider_class = _PROVIDERS[provider]
        instance = provider_class(model=model, **kwargs)

        # Cache if no custom kwargs
        if not kwargs:
            self._cache[cache_key] = instance

        return instance

    def get_minimax(self, model: str | None = None, **kwargs) -> MinimaxProvider:
        """Get a Minimax provider instance."""
        return self.get_provider("minimax", model, **kwargs)  # type: ignore

    def get_qwen(self, model: str | None = None, **kwargs) -> QwenProvider:
        """Get a Qwen provider instance."""
        return self.get_provider("qwen", model, **kwargs)  # type: ignore

    def clear_cache(self) -> None:
        """Clear the provider cache."""
        self._cache.clear()


# Global router instance
_router: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    """Get the global LLM router instance."""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def get_provider(
    provider: ProviderType | None = None,
    model: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """Convenience function to get an LLM provider.

    Args:
        provider: Provider type ("minimax", "qwen").
        model: Optional model override.
        **kwargs: Additional provider arguments.

    Returns:
        An LLM provider instance.
    """
    return get_llm_router().get_provider(provider, model, **kwargs)
