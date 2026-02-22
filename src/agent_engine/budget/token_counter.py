"""Token counting utilities."""

from typing import Any

import tiktoken


class TokenCounter:
    """Token counter for various LLM models."""

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        # Minimax models (using cl100k_base as approximation)
        "abab6.5-chat": "cl100k_base",
        "abab6.5s-chat": "cl100k_base",
        "abab6.5-pro": "cl100k_base",
        # Minimax models
        "abab6.5-chat": "cl100k_base",
        "abab6.5s-chat": "cl100k_base",
        "abab6.5-pro": "cl100k_base",
    }

    # Per-message overhead for chat models
    MESSAGE_OVERHEAD = {
        "abab6.5-chat": 3,
        "abab6.5s-chat": 3,
        "abab6.5-pro": 3,
    }

    def __init__(self, model: str = "abab6.5-chat"):
        """Initialize token counter.

        Args:
            model: Model name for encoding selection.
        """
        self.model = model
        self._encoding_name = self._get_encoding_name(model)
        self._encoding = tiktoken.get_encoding(self._encoding_name)

    def _get_encoding_name(self, model: str) -> str:
        """Get encoding name for a model."""
        # Check exact match
        if model in self.MODEL_ENCODINGS:
            return self.MODEL_ENCODINGS[model]

        # Check prefix match
        for prefix, encoding in self.MODEL_ENCODINGS.items():
            if model.startswith(prefix):
                return encoding

        # Default to cl100k_base
        return "cl100k_base"

    def count_text(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_messages(
        self,
        messages: list[dict[str, Any]],
        include_overhead: bool = True,
    ) -> int:
        """Count tokens in a list of messages.

        Args:
            messages: List of message dictionaries.
            include_overhead: Include per-message overhead.

        Returns:
            Total token count.
        """
        total = 0
        overhead = self.MESSAGE_OVERHEAD.get(self.model, 3) if include_overhead else 0

        for message in messages:
            # Count role
            total += self.count_text(message.get("role", ""))

            # Count content
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                # Handle multi-modal content
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += self.count_text(block.get("text", ""))

            # Count tool calls
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    total += self.count_text(tc.get("name", ""))
                    args = tc.get("args", {})
                    if isinstance(args, str):
                        total += self.count_text(args)
                    else:
                        import json
                        total += self.count_text(json.dumps(args))

            # Add overhead
            total += overhead

        # Add reply priming tokens
        total += 3

        return total

    def count_tool_schemas(self, tools: list[dict[str, Any]]) -> int:
        """Count tokens in tool schemas.

        Args:
            tools: List of tool schema dictionaries.

        Returns:
            Token count for tool schemas.
        """
        import json

        total = 0
        for tool in tools:
            # Count the full tool definition
            total += self.count_text(json.dumps(tool))

        return total

    def truncate_text(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "...",
    ) -> str:
        """Truncate text to a maximum number of tokens.

        Args:
            text: The text to truncate.
            max_tokens: Maximum tokens allowed.
            suffix: Suffix to add when truncated.

        Returns:
            Truncated text.
        """
        tokens = self._encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        suffix_tokens = self._encoding.encode(suffix)
        truncated_tokens = tokens[: max_tokens - len(suffix_tokens)]

        return self._encoding.decode(truncated_tokens) + suffix

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        """Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name (uses counter's model if not specified).

        Returns:
            Estimated cost in USD.
        """
        model = model or self.model

        # Pricing per 1M tokens (approximate, may be outdated)
        PRICING = {
            "abab6.5-chat": {"input": 0.01, "output": 0.01},  # Approximate pricing
            "abab6.5s-chat": {"input": 0.005, "output": 0.005},
            "abab6.5-pro": {"input": 0.02, "output": 0.02},
        }

        pricing = PRICING.get(model, {"input": 0.01, "output": 0.01})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


# Global counter cache
_counters: dict[str, TokenCounter] = {}


def get_token_counter(model: str = "abab6.5-chat") -> TokenCounter:
    """Get or create a token counter for a model.

    Args:
        model: The model name.

    Returns:
        TokenCounter instance.
    """
    if model not in _counters:
        _counters[model] = TokenCounter(model)
    return _counters[model]
