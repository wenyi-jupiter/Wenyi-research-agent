"""Tests for tool registry."""

import pytest

from agent_engine.tools import (
    MCPToolRequest,
    MCPToolSchema,
    MCPParameterSchema,
    ToolRegistry,
    tool,
)


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_decorator(self):
        """Test registering a tool with decorator."""
        registry = ToolRegistry()

        @registry.register(
            name="test_tool",
            description="A test tool",
            tags=["test"],
        )
        async def test_tool(arg1: str, arg2: int = 10) -> str:
            return f"{arg1}-{arg2}"

        assert "test_tool" in registry
        assert len(registry) == 1

        schema = registry.get_tool("test_tool")
        assert schema is not None
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert "test" in schema.tags

    def test_register_tool_explicit(self):
        """Test registering a tool with explicit schema."""
        registry = ToolRegistry()

        schema = MCPToolSchema(
            name="explicit_tool",
            description="Explicitly registered tool",
            parameters=MCPParameterSchema(
                type="object",
                properties={"input": {"type": "string"}},
                required=["input"],
            ),
        )

        async def impl(input: str) -> str:
            return f"result: {input}"

        registry.register_tool(schema, impl)

        assert "explicit_tool" in registry
        assert registry.get_tool("explicit_tool") == schema

    def test_list_tools_filter_by_tags(self):
        """Test listing tools filtered by tags."""
        registry = ToolRegistry()

        @registry.register(name="tool_a", description="Tool A", tags=["alpha", "common"])
        async def tool_a():
            pass

        @registry.register(name="tool_b", description="Tool B", tags=["beta", "common"])
        async def tool_b():
            pass

        @registry.register(name="tool_c", description="Tool C", tags=["gamma"])
        async def tool_c():
            pass

        # Filter by single tag
        common_tools = registry.list_tools(tags=["common"])
        assert len(common_tools) == 2

        alpha_tools = registry.list_tools(tags=["alpha"])
        assert len(alpha_tools) == 1
        assert alpha_tools[0].name == "tool_a"

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        @registry.register(name="removable", description="Will be removed")
        async def removable():
            pass

        assert "removable" in registry

        result = registry.unregister("removable")
        assert result is True
        assert "removable" not in registry

        # Try to unregister non-existent tool
        result = registry.unregister("non_existent")
        assert result is False

    def test_get_langchain_tools(self):
        """Test getting LangChain tools."""
        registry = ToolRegistry()

        @registry.register(name="lc_tool", description="LangChain compatible tool")
        async def lc_tool(query: str) -> str:
            return f"Result for {query}"

        lc_tools = registry.get_langchain_tools()
        assert len(lc_tools) == 1
        assert lc_tools[0].name == "lc_tool"

    def test_auto_parameter_extraction(self):
        """Test automatic parameter extraction from function signature."""
        registry = ToolRegistry()

        @registry.register(name="auto_params", description="Auto params test")
        async def auto_params(
            required_str: str,
            optional_int: int = 42,
            optional_bool: bool = True,
        ) -> dict:
            return {"str": required_str, "int": optional_int, "bool": optional_bool}

        schema = registry.get_tool("auto_params")
        assert schema is not None

        params = schema.parameters
        assert "required_str" in params.properties
        assert "optional_int" in params.properties
        assert "optional_bool" in params.properties
        assert "required_str" in params.required
        assert "optional_int" not in params.required


class TestGlobalRegistry:
    """Tests for global tool decorator."""

    def test_global_tool_decorator(self):
        """Test the global @tool decorator."""
        from agent_engine.tools.registry import get_tool_registry

        @tool(name="global_test", description="Global test tool", tags=["global"])
        async def global_test(value: str) -> str:
            return value.upper()

        registry = get_tool_registry()
        assert "global_test" in registry
