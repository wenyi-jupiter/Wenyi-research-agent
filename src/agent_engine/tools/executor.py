"""Concurrent tool executor with retry logic."""

import asyncio
import inspect
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)

from agent_engine.config import get_settings
from agent_engine.tools.mcp_protocol import (
    MCPToolRequest,
    MCPToolResult,
    MCPToolResultStatus,
)
from agent_engine.tools.registry import ToolRegistry, get_tool_registry


@dataclass
class ExecutionConfig:
    """Configuration for tool execution."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 16.0
    timeout: float = 30.0
    max_concurrent: int = 10


@dataclass
class ExecutionStats:
    """Statistics for tool executions."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    total_time_ms: float = 0.0
    tool_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, tool_name: str, result: MCPToolResult, retries: int = 0) -> None:
        """Record execution statistics."""
        self.total_calls += 1
        self.total_time_ms += result.execution_time_ms

        if result.is_success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

        if retries > 0:
            self.retried_calls += retries

        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0.0,
            }

        stats = self.tool_stats[tool_name]
        stats["calls"] += 1
        stats["total_time_ms"] += result.execution_time_ms
        if result.is_success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1


class CircuitBreaker:
    """Simple circuit breaker for failing tools."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures: dict[str, int] = {}
        self._open_time: dict[str, float] = {}

    def record_success(self, tool_name: str) -> None:
        """Record a successful execution."""
        self._failures[tool_name] = 0
        self._open_time.pop(tool_name, None)

    def record_failure(self, tool_name: str) -> None:
        """Record a failed execution."""
        self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
        if self._failures[tool_name] >= self.failure_threshold:
            self._open_time[tool_name] = time.time()

    def is_open(self, tool_name: str) -> bool:
        """Check if circuit is open (tool should be blocked)."""
        if tool_name not in self._open_time:
            return False

        elapsed = time.time() - self._open_time[tool_name]
        if elapsed >= self.recovery_timeout:
            # Half-open: allow one request to test
            self._open_time.pop(tool_name)
            self._failures[tool_name] = self.failure_threshold - 1
            return False

        return True

    def reset(self, tool_name: str | None = None) -> None:
        """Reset circuit breaker state."""
        if tool_name:
            self._failures.pop(tool_name, None)
            self._open_time.pop(tool_name, None)
        else:
            self._failures.clear()
            self._open_time.clear()


class ToolExecutor:
    """Executor for running tools with retry and concurrency support."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        config: ExecutionConfig | None = None,
    ):
        """Initialize the executor.

        Args:
            registry: Tool registry to use.
            config: Execution configuration.
        """
        self.registry = registry or get_tool_registry()
        self.config = config or self._default_config()
        self.stats = ExecutionStats()
        self.circuit_breaker = CircuitBreaker()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    def _default_config(self) -> ExecutionConfig:
        """Get default configuration from settings."""
        settings = get_settings()
        return ExecutionConfig(
            max_retries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            timeout=settings.tool_timeout,
        )

    async def execute(
        self,
        request: MCPToolRequest,
        skip_retry: bool = False,
    ) -> MCPToolResult:
        """Execute a single tool request.

        Args:
            request: The tool request.
            skip_retry: Skip retry logic.

        Returns:
            The execution result.
        """
        tool_name = request.tool_name
        start_time = time.time()

        # Check if tool exists
        impl = self.registry.get_implementation(tool_name)
        if impl is None:
            return MCPToolResult(
                tool_name=tool_name,
                status=MCPToolResultStatus.ERROR,
                error=f"Tool not found: {tool_name}",
                request_id=request.request_id,
            )

        # Check circuit breaker
        if self.circuit_breaker.is_open(tool_name):
            return MCPToolResult(
                tool_name=tool_name,
                status=MCPToolResultStatus.ERROR,
                error=f"Tool circuit open: {tool_name}",
                request_id=request.request_id,
            )

        retries = 0
        last_error: str | None = None

        async with self._semaphore:
            if skip_retry:
                result = await self._execute_once(impl, request, start_time)
            else:
                try:
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(self.config.max_retries),
                        wait=wait_exponential(
                            multiplier=self.config.base_delay,
                            max=self.config.max_delay,
                        ),
                        reraise=True,
                    ):
                        with attempt:
                            result = await self._execute_once(impl, request, start_time)
                            if not result.is_success:
                                retries += 1
                                last_error = result.error
                                raise RuntimeError(result.error)

                except RetryError:
                    result = MCPToolResult(
                        tool_name=tool_name,
                        status=MCPToolResultStatus.ERROR,
                        error=f"Max retries exceeded: {last_error}",
                        execution_time_ms=(time.time() - start_time) * 1000,
                        request_id=request.request_id,
                    )

        # Update circuit breaker
        if result.is_success:
            self.circuit_breaker.record_success(tool_name)
        else:
            self.circuit_breaker.record_failure(tool_name)

        # Record stats
        self.stats.record(tool_name, result, retries)

        return result

    async def _execute_once(
        self,
        impl: Any,
        request: MCPToolRequest,
        start_time: float,
    ) -> MCPToolResult:
        """Execute a tool once without retry."""
        tool_name = request.tool_name

        try:
            # Execute with timeout
            if inspect.iscoroutinefunction(impl):
                result = await asyncio.wait_for(
                    impl(**request.arguments),
                    timeout=self.config.timeout,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(impl, **request.arguments),
                    timeout=self.config.timeout,
                )

            return MCPToolResult(
                tool_name=tool_name,
                status=MCPToolResultStatus.SUCCESS,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
                request_id=request.request_id,
            )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=tool_name,
                status=MCPToolResultStatus.TIMEOUT,
                error=f"Tool execution timed out after {self.config.timeout}s",
                execution_time_ms=(time.time() - start_time) * 1000,
                request_id=request.request_id,
            )

        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                status=MCPToolResultStatus.ERROR,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                request_id=request.request_id,
            )

    async def execute_batch(
        self,
        requests: Sequence[MCPToolRequest],
        parallel: bool = True,
    ) -> list[MCPToolResult]:
        """Execute multiple tool requests.

        Args:
            requests: List of tool requests.
            parallel: Execute in parallel (default) or sequentially.

        Returns:
            List of execution results.
        """
        if not requests:
            return []

        if parallel:
            tasks = [self.execute(req) for req in requests]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for req in requests:
                result = await self.execute(req)
                results.append(result)
            return results

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.stats = ExecutionStats()

    def reset_circuit_breaker(self, tool_name: str | None = None) -> None:
        """Reset circuit breaker state."""
        self.circuit_breaker.reset(tool_name)


# Global executor instance
_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    """Get the global tool executor instance."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor
