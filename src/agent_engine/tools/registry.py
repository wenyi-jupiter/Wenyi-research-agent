"""Tool registry for managing and discovering tools."""

import inspect
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from langchain_core.tools import BaseTool, StructuredTool

from agent_engine.tools.mcp_protocol import (
    MCPCapabilities,
    MCPParameterSchema,
    MCPServerInfo,
    MCPToolSchema,
    MCPToolType,
)


P = ParamSpec("P")
R = TypeVar("R")

ToolFunction = Callable[P, R] | Callable[P, Coroutine[Any, Any, R]]


class ToolRegistry:
    """Registry for managing MCP-compatible tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: dict[str, MCPToolSchema] = {}
        self._implementations: dict[str, ToolFunction] = {}
        self._langchain_tools: dict[str, BaseTool] = {}
        self._tags_index: dict[str, set[str]] = {}

    @property
    def server_info(self) -> MCPServerInfo:
        """Get MCP server information."""
        return MCPServerInfo(
            name="agent-engine-tools",
            version="1.0.0",
            capabilities=MCPCapabilities(tools=True),
        )

    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        parameters: MCPParameterSchema | dict[str, Any] | None = None,
        tags: list[str] | None = None,
        tool_type: MCPToolType = MCPToolType.FUNCTION,
    ) -> Callable[[ToolFunction], ToolFunction]:
        """Decorator to register a tool function.

        Args:
            name: Tool name (defaults to function name).
            description: Tool description (defaults to docstring).
            parameters: Parameter schema (auto-generated if not provided).
            tags: Tags for categorization.
            tool_type: Type of tool.

        Returns:
            Decorator function.
        """

        def decorator(func: ToolFunction) -> ToolFunction:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or f"Execute {tool_name}"

            # Auto-generate parameter schema from function signature
            if parameters is None:
                param_schema = self._extract_parameters(func)
            elif isinstance(parameters, dict):
                param_schema = MCPParameterSchema(**parameters)
            else:
                param_schema = parameters

            # Create tool schema
            schema = MCPToolSchema(
                name=tool_name,
                description=tool_desc.strip(),
                parameters=param_schema,
                tool_type=tool_type,
                tags=tags or [],
            )

            # Register the tool
            self._tools[tool_name] = schema
            self._implementations[tool_name] = func

            # Create LangChain tool
            self._langchain_tools[tool_name] = StructuredTool.from_function(
                func=func,
                name=tool_name,
                description=tool_desc.strip(),
                coroutine=func if inspect.iscoroutinefunction(func) else None,
            )

            # Update tags index
            for tag in schema.tags:
                if tag not in self._tags_index:
                    self._tags_index[tag] = set()
                self._tags_index[tag].add(tool_name)

            return func

        return decorator

    def register_tool(
        self,
        schema: MCPToolSchema,
        implementation: ToolFunction,
    ) -> None:
        """Register a tool with explicit schema and implementation.

        Args:
            schema: The tool schema.
            implementation: The tool function.
        """
        self._tools[schema.name] = schema
        self._implementations[schema.name] = implementation

        # Create LangChain tool
        self._langchain_tools[schema.name] = StructuredTool.from_function(
            func=implementation,
            name=schema.name,
            description=schema.description,
            coroutine=implementation if inspect.iscoroutinefunction(implementation) else None,
        )

        # Update tags index
        for tag in schema.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(schema.name)

    def register_langchain_tool(self, tool: BaseTool, tags: list[str] | None = None) -> None:
        """Register an existing LangChain tool.

        Args:
            tool: The LangChain tool to register.
            tags: Optional tags for categorization.
        """
        # Create MCP schema from LangChain tool
        param_schema = MCPParameterSchema()
        if hasattr(tool, "args_schema") and tool.args_schema:
            schema_dict = tool.args_schema.model_json_schema()
            param_schema = MCPParameterSchema(
                type="object",
                properties=schema_dict.get("properties", {}),
                required=schema_dict.get("required", []),
            )

        schema = MCPToolSchema(
            name=tool.name,
            description=tool.description or f"Execute {tool.name}",
            parameters=param_schema,
            tags=tags or [],
        )

        self._tools[tool.name] = schema
        self._langchain_tools[tool.name] = tool

        # Store the run method as implementation
        if hasattr(tool, "_arun"):
            self._implementations[tool.name] = tool._arun
        else:
            self._implementations[tool.name] = tool._run

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: The tool name.

        Returns:
            True if tool was unregistered, False if not found.
        """
        if name not in self._tools:
            return False

        schema = self._tools.pop(name)
        self._implementations.pop(name, None)
        self._langchain_tools.pop(name, None)

        # Update tags index
        for tag in schema.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(name)

        return True

    def get_tool(self, name: str) -> MCPToolSchema | None:
        """Get a tool schema by name."""
        return self._tools.get(name)

    def get_implementation(self, name: str) -> ToolFunction | None:
        """Get a tool implementation by name."""
        return self._implementations.get(name)

    def get_langchain_tool(self, name: str) -> BaseTool | None:
        """Get a LangChain tool by name."""
        return self._langchain_tools.get(name)

    def list_tools(
        self,
        tags: list[str] | None = None,
        tool_type: MCPToolType | None = None,
    ) -> list[MCPToolSchema]:
        """List all registered tools.

        Args:
            tags: Filter by tags (OR logic).
            tool_type: Filter by tool type.

        Returns:
            List of tool schemas.
        """
        tools = list(self._tools.values())

        if tags:
            matching_names: set[str] = set()
            for tag in tags:
                matching_names.update(self._tags_index.get(tag, set()))
            tools = [t for t in tools if t.name in matching_names]

        if tool_type:
            tools = [t for t in tools if t.tool_type == tool_type]

        return tools

    def get_langchain_tools(
        self,
        names: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[BaseTool]:
        """Get LangChain tools for use with agents.

        Args:
            names: Specific tool names to get.
            tags: Filter by tags.

        Returns:
            List of LangChain tools.
        """
        if names:
            return [
                self._langchain_tools[n]
                for n in names
                if n in self._langchain_tools
            ]

        if tags:
            schemas = self.list_tools(tags=tags)
            return [
                self._langchain_tools[s.name]
                for s in schemas
                if s.name in self._langchain_tools
            ]

        return list(self._langchain_tools.values())

    def _extract_parameters(self, func: ToolFunction) -> MCPParameterSchema:
        """Extract parameter schema from function signature."""
        sig = inspect.signature(func)
        hints = getattr(func, "__annotations__", {})

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, Any)
            type_str = self._python_type_to_json(param_type)

            properties[param_name] = {"type": type_str}

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return MCPParameterSchema(
            type="object",
            properties=properties,
            required=required,
        )

    def _python_type_to_json(self, python_type: type) -> str:
        """Convert Python type to JSON Schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        origin = getattr(python_type, "__origin__", None)
        if origin is not None:
            python_type = origin

        return type_map.get(python_type, "string")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global registry instance
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: MCPParameterSchema | dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Callable[[ToolFunction], ToolFunction]:
    """Convenience decorator to register a tool to the global registry.

    Args:
        name: Tool name.
        description: Tool description.
        parameters: Parameter schema.
        tags: Tags for categorization.

    Returns:
        Decorator function.
    """
    registry = get_tool_registry()
    return registry.register(
        name=name,
        description=description,
        parameters=parameters,
        tags=tags,
    )
