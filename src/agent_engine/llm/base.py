"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", 0) or self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", 0) or self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    content: str = ""
    tool_call_chunk: dict[str, Any] | None = None
    finish_reason: str | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, **kwargs: Any):
        """Initialize the provider.

        Args:
            model: The model name to use.
            **kwargs: Additional provider-specific arguments.
        """
        self.model = model
        self._kwargs = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        ...

    @abstractmethod
    def get_langchain_model(self, tools: Sequence[BaseTool] | None = None) -> BaseChatModel:
        """Get a LangChain chat model instance.

        Args:
            tools: Optional tools to bind to the model.

        Returns:
            A LangChain BaseChatModel instance.
        """
        ...

    @abstractmethod
    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke the LLM with messages.

        Args:
            messages: The conversation messages.
            tools: Optional tools available for the model.
            **kwargs: Additional arguments.

        Returns:
            The LLM response.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream responses from the LLM.

        Args:
            messages: The conversation messages.
            tools: Optional tools available for the model.
            **kwargs: Additional arguments.

        Yields:
            Stream chunks as they arrive.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
