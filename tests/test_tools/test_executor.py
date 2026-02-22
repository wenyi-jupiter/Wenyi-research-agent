"""Tests for tool executor."""

import asyncio

import pytest

from agent_engine.tools import (
    ExecutionConfig,
    MCPToolRequest,
    MCPToolResultStatus,
    ToolExecutor,
    ToolRegistry,
)


@pytest.fixture
def registry():
    """Create a test registry with sample tools."""
    reg = ToolRegistry()

    @reg.register(name="fast_tool", description="A fast tool")
    async def fast_tool(value: str) -> str:
        return f"fast: {value}"

    @reg.register(name="slow_tool", description="A slow tool")
    async def slow_tool(delay: float = 0.1) -> str:
        await asyncio.sleep(delay)
        return f"slow after {delay}s"

    @reg.register(name="failing_tool", description="A tool that fails")
    async def failing_tool(should_fail: bool = True) -> str:
        if should_fail:
            raise ValueError("Intentional failure")
        return "success"

    return reg


@pytest.fixture
def executor(registry):
    """Create a test executor."""
    config = ExecutionConfig(
        max_retries=2,
        base_delay=0.1,
        timeout=5.0,
        max_concurrent=5,
    )
    return ToolExecutor(registry=registry, config=config)


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        """Test successful tool execution."""
        request = MCPToolRequest(
            tool_name="fast_tool",
            arguments={"value": "hello"},
        )

        result = await executor.execute(request)

        assert result.is_success
        assert result.status == MCPToolResultStatus.SUCCESS
        assert result.result == "fast: hello"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, executor):
        """Test execution of unknown tool."""
        request = MCPToolRequest(
            tool_name="unknown_tool",
            arguments={},
        )

        result = await executor.execute(request)

        assert not result.is_success
        assert result.status == MCPToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_retry(self, registry):
        """Test execution with retry on failure."""
        call_count = 0

        @registry.register(name="flaky_tool", description="Sometimes fails")
        async def flaky_tool() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Temporary failure")
            return "success after retry"

        config = ExecutionConfig(max_retries=3, base_delay=0.01)
        executor = ToolExecutor(registry=registry, config=config)

        request = MCPToolRequest(tool_name="flaky_tool", arguments={})
        result = await executor.execute(request)

        # Should succeed after retry
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_execute_timeout(self, registry):
        """Test execution timeout."""
        @registry.register(name="very_slow_tool", description="Takes forever")
        async def very_slow_tool() -> str:
            await asyncio.sleep(10)
            return "done"

        config = ExecutionConfig(timeout=0.1, max_retries=1)
        executor = ToolExecutor(registry=registry, config=config)

        request = MCPToolRequest(tool_name="very_slow_tool", arguments={})
        result = await executor.execute(request, skip_retry=True)

        assert result.status == MCPToolResultStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_batch_parallel(self, executor):
        """Test parallel batch execution."""
        requests = [
            MCPToolRequest(tool_name="slow_tool", arguments={"delay": 0.1}),
            MCPToolRequest(tool_name="slow_tool", arguments={"delay": 0.1}),
            MCPToolRequest(tool_name="slow_tool", arguments={"delay": 0.1}),
        ]

        import time
        start = time.time()
        results = await executor.execute_batch(requests, parallel=True)
        elapsed = time.time() - start

        assert len(results) == 3
        assert all(r.is_success for r in results)
        # Parallel execution should be faster than sequential
        assert elapsed < 0.25  # 3 * 0.1s sequential would be 0.3s

    @pytest.mark.asyncio
    async def test_execute_batch_sequential(self, executor):
        """Test sequential batch execution."""
        requests = [
            MCPToolRequest(tool_name="fast_tool", arguments={"value": "1"}),
            MCPToolRequest(tool_name="fast_tool", arguments={"value": "2"}),
        ]

        results = await executor.execute_batch(requests, parallel=False)

        assert len(results) == 2
        assert results[0].result == "fast: 1"
        assert results[1].result == "fast: 2"

    @pytest.mark.asyncio
    async def test_execution_stats(self, executor):
        """Test execution statistics tracking."""
        executor.reset_stats()

        request = MCPToolRequest(tool_name="fast_tool", arguments={"value": "test"})
        await executor.execute(request)

        stats = executor.stats
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0
        assert "fast_tool" in stats.tool_stats

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, registry):
        """Test circuit breaker functionality."""
        @registry.register(name="always_fails", description="Always fails")
        async def always_fails() -> str:
            raise RuntimeError("Always fails")

        config = ExecutionConfig(max_retries=1, base_delay=0.01)
        executor = ToolExecutor(registry=registry, config=config)
        executor.circuit_breaker.failure_threshold = 2
        executor.circuit_breaker.recovery_timeout = 1.0

        # First few calls should attempt
        request = MCPToolRequest(tool_name="always_fails", arguments={})
        await executor.execute(request, skip_retry=True)
        await executor.execute(request, skip_retry=True)

        # Circuit should now be open
        assert executor.circuit_breaker.is_open("always_fails")

        # Next call should fail fast
        result = await executor.execute(request, skip_retry=True)
        assert "circuit open" in result.error.lower()
