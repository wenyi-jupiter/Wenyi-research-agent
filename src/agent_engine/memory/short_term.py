"""Short-term memory for conversation context."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.messages import BaseMessage


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """Short-term memory for maintaining conversation context.

    This class manages:
    - Recent conversation turns
    - Working memory for current task
    - Token-aware context truncation
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens: int = 8000,
    ):
        """Initialize short-term memory.

        Args:
            max_turns: Maximum conversation turns to keep.
            max_tokens: Maximum tokens in context.
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._working_memory: dict[str, Any] = {}
        self._messages: list[BaseMessage] = []

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a conversation turn.

        Args:
            role: The role (user, assistant, system, tool).
            content: The message content.
            metadata: Optional metadata.
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._turns.append(turn)

    def add_message(self, message: BaseMessage) -> None:
        """Add a LangChain message.

        Args:
            message: The message to add.
        """
        self._messages.append(message)

        # Keep messages bounded
        while len(self._messages) > self.max_turns * 2:
            self._messages.pop(0)

    def get_messages(self, limit: int | None = None) -> list[BaseMessage]:
        """Get recent messages.

        Args:
            limit: Maximum messages to return.

        Returns:
            List of messages.
        """
        if limit:
            return self._messages[-limit:]
        return list(self._messages)

    def get_turns(self, limit: int | None = None) -> list[ConversationTurn]:
        """Get recent conversation turns.

        Args:
            limit: Maximum turns to return.

        Returns:
            List of conversation turns.
        """
        turns = list(self._turns)
        if limit:
            return turns[-limit:]
        return turns

    def get_context(
        self,
        max_chars: int | None = None,
        include_system: bool = False,
    ) -> str:
        """Get formatted conversation context.

        Args:
            max_chars: Maximum characters in context.
            include_system: Include system messages.

        Returns:
            Formatted context string.
        """
        turns = self.get_turns()

        if not include_system:
            turns = [t for t in turns if t.role != "system"]

        context_parts = []
        total_chars = 0
        max_chars = max_chars or (self.max_tokens * 4)

        # Build context from most recent turns
        for turn in reversed(turns):
            formatted = f"[{turn.role}]: {turn.content}"

            if total_chars + len(formatted) > max_chars:
                break

            context_parts.insert(0, formatted)
            total_chars += len(formatted)

        return "\n\n".join(context_parts)

    def set_working(self, key: str, value: Any) -> None:
        """Set a value in working memory.

        Args:
            key: The key to set.
            value: The value to store.
        """
        self._working_memory[key] = value

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory.

        Args:
            key: The key to get.
            default: Default value if not found.

        Returns:
            The stored value or default.
        """
        return self._working_memory.get(key, default)

    def clear_working(self, key: str | None = None) -> None:
        """Clear working memory.

        Args:
            key: Specific key to clear, or all if None.
        """
        if key:
            self._working_memory.pop(key, None)
        else:
            self._working_memory.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of current memory state.

        Returns:
            Dictionary with memory statistics.
        """
        return {
            "turn_count": len(self._turns),
            "message_count": len(self._messages),
            "working_memory_keys": list(self._working_memory.keys()),
            "total_chars": sum(len(t.content) for t in self._turns),
        }

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._turns.clear()
        self._messages.clear()
        self._working_memory.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat(),
                    "metadata": t.metadata,
                }
                for t in self._turns
            ],
            "working_memory": self._working_memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs) -> "ShortTermMemory":
        """Create from dictionary.

        Args:
            data: Dictionary representation.
            **kwargs: Additional constructor arguments.

        Returns:
            ShortTermMemory instance.
        """
        memory = cls(**kwargs)

        for turn_data in data.get("turns", []):
            memory.add_turn(
                role=turn_data["role"],
                content=turn_data["content"],
                metadata=turn_data.get("metadata"),
            )

        memory._working_memory = data.get("working_memory", {})

        return memory
