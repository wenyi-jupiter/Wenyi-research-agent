"""Execution limits and budget management."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class BudgetExceededReason(str, Enum):
    """Reason for budget being exceeded."""

    TOKEN_LIMIT = "token_limit"
    STEP_LIMIT = "step_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    TIMEOUT = "timeout"
    COST_LIMIT = "cost_limit"


@dataclass
class ExecutionLimits:
    """Execution limits configuration."""

    max_tokens: int = 100_000
    max_steps: int = 50
    max_tool_calls: int = 100
    timeout_seconds: int = 600
    max_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "timeout_seconds": self.timeout_seconds,
            "max_cost_usd": self.max_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionLimits":
        """Create from dictionary."""
        return cls(
            max_tokens=data.get("max_tokens", 100_000),
            max_steps=data.get("max_steps", 50),
            max_tool_calls=data.get("max_tool_calls", 100),
            timeout_seconds=data.get("timeout_seconds", 600),
            max_cost_usd=data.get("max_cost_usd"),
        )


@dataclass
class BudgetUsage:
    """Current budget usage tracking."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    step_count: int = 0
    tool_call_count: int = 0
    estimated_cost_usd: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage.

        Args:
            input_tokens: Input tokens used.
            output_tokens: Output tokens used.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens

    def increment_step(self) -> None:
        """Increment step counter."""
        self.step_count += 1

    def increment_tool_calls(self, count: int = 1) -> None:
        """Increment tool call counter."""
        self.tool_call_count += count

    def add_cost(self, cost: float) -> None:
        """Add to estimated cost."""
        self.estimated_cost_usd += cost

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "step_count": self.step_count,
            "tool_call_count": self.tool_call_count,
            "estimated_cost_usd": self.estimated_cost_usd,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class BudgetCheckResult:
    """Result of a budget check."""

    is_exceeded: bool
    reason: BudgetExceededReason | None = None
    message: str = ""
    usage: BudgetUsage | None = None
    limits: ExecutionLimits | None = None

    @property
    def can_continue(self) -> bool:
        """Check if execution can continue."""
        return not self.is_exceeded


class BudgetManager:
    """Manager for tracking and enforcing execution budgets."""

    def __init__(
        self,
        limits: ExecutionLimits | None = None,
    ):
        """Initialize budget manager.

        Args:
            limits: Execution limits to enforce.
        """
        self.limits = limits or ExecutionLimits()
        self.usage = BudgetUsage()
        self._callbacks: list[callable] = []

    def reset(self) -> None:
        """Reset usage tracking."""
        self.usage = BudgetUsage()

    def record_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
    ) -> BudgetCheckResult:
        """Record an LLM API call.

        Args:
            input_tokens: Input tokens used.
            output_tokens: Output tokens used.
            cost: Estimated cost.

        Returns:
            Budget check result.
        """
        self.usage.add_tokens(input_tokens, output_tokens)
        self.usage.increment_step()
        self.usage.add_cost(cost)

        return self.check()

    def record_tool_call(self, count: int = 1) -> BudgetCheckResult:
        """Record tool call(s).

        Args:
            count: Number of tool calls.

        Returns:
            Budget check result.
        """
        self.usage.increment_tool_calls(count)
        return self.check()

    def check(self) -> BudgetCheckResult:
        """Check if budget limits are exceeded.

        Returns:
            Budget check result.
        """
        # Check token limit
        if self.usage.total_tokens > self.limits.max_tokens:
            return BudgetCheckResult(
                is_exceeded=True,
                reason=BudgetExceededReason.TOKEN_LIMIT,
                message=f"Token limit exceeded: {self.usage.total_tokens} > {self.limits.max_tokens}",
                usage=self.usage,
                limits=self.limits,
            )

        # Check step limit
        if self.usage.step_count > self.limits.max_steps:
            return BudgetCheckResult(
                is_exceeded=True,
                reason=BudgetExceededReason.STEP_LIMIT,
                message=f"Step limit exceeded: {self.usage.step_count} > {self.limits.max_steps}",
                usage=self.usage,
                limits=self.limits,
            )

        # Check tool call limit
        if self.usage.tool_call_count > self.limits.max_tool_calls:
            return BudgetCheckResult(
                is_exceeded=True,
                reason=BudgetExceededReason.TOOL_CALL_LIMIT,
                message=f"Tool call limit exceeded: {self.usage.tool_call_count} > {self.limits.max_tool_calls}",
                usage=self.usage,
                limits=self.limits,
            )

        # Check timeout
        if self.usage.elapsed_seconds > self.limits.timeout_seconds:
            return BudgetCheckResult(
                is_exceeded=True,
                reason=BudgetExceededReason.TIMEOUT,
                message=f"Timeout exceeded: {self.usage.elapsed_seconds:.1f}s > {self.limits.timeout_seconds}s",
                usage=self.usage,
                limits=self.limits,
            )

        # Check cost limit
        if (
            self.limits.max_cost_usd is not None
            and self.usage.estimated_cost_usd > self.limits.max_cost_usd
        ):
            return BudgetCheckResult(
                is_exceeded=True,
                reason=BudgetExceededReason.COST_LIMIT,
                message=f"Cost limit exceeded: ${self.usage.estimated_cost_usd:.4f} > ${self.limits.max_cost_usd:.4f}",
                usage=self.usage,
                limits=self.limits,
            )

        # All checks passed
        return BudgetCheckResult(
            is_exceeded=False,
            usage=self.usage,
            limits=self.limits,
        )

    def get_remaining(self) -> dict[str, Any]:
        """Get remaining budget.

        Returns:
            Dictionary with remaining budget values.
        """
        return {
            "tokens": max(0, self.limits.max_tokens - self.usage.total_tokens),
            "steps": max(0, self.limits.max_steps - self.usage.step_count),
            "tool_calls": max(0, self.limits.max_tool_calls - self.usage.tool_call_count),
            "time_seconds": max(
                0, self.limits.timeout_seconds - self.usage.elapsed_seconds
            ),
            "cost_usd": (
                max(0, self.limits.max_cost_usd - self.usage.estimated_cost_usd)
                if self.limits.max_cost_usd
                else None
            ),
        }

    def get_utilization(self) -> dict[str, float]:
        """Get budget utilization percentages.

        Returns:
            Dictionary with utilization percentages (0-100).
        """
        return {
            "tokens": (self.usage.total_tokens / self.limits.max_tokens) * 100,
            "steps": (self.usage.step_count / self.limits.max_steps) * 100,
            "tool_calls": (self.usage.tool_call_count / self.limits.max_tool_calls) * 100,
            "time": (self.usage.elapsed_seconds / self.limits.timeout_seconds) * 100,
        }

    def on_budget_exceeded(self, callback: callable) -> None:
        """Register a callback for when budget is exceeded.

        Args:
            callback: Function to call with BudgetCheckResult.
        """
        self._callbacks.append(callback)

    def _notify_exceeded(self, result: BudgetCheckResult) -> None:
        """Notify callbacks of budget exceeded."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception:
                pass  # Don't let callback errors affect execution
