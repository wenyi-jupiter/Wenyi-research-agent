"""Model Context Protocol (MCP) implementation for tool management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MCPToolType(str, Enum):
    """Tool types in MCP."""

    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT = "prompt"


class MCPParameterSchema(BaseModel):
    """JSON Schema for a tool parameter."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    description: str | None = None


class MCPToolSchema(BaseModel):
    """Schema definition for an MCP tool."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Tool description")
    parameters: MCPParameterSchema = Field(
        default_factory=MCPParameterSchema,
        description="JSON Schema for parameters",
    )
    tool_type: MCPToolType = Field(default=MCPToolType.FUNCTION)
    tags: list[str] = Field(default_factory=list)
    version: str = Field(default="1.0.0")

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": self.parameters.type,
                    "properties": self.parameters.properties,
                    "required": self.parameters.required,
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": self.parameters.type,
                "properties": self.parameters.properties,
                "required": self.parameters.required,
            },
        }


class MCPToolRequest(BaseModel):
    """Request to execute a tool."""

    tool_name: str = Field(..., description="Name of tool to execute")
    arguments: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = Field(default=None, description="Optional request ID")


class MCPToolResultStatus(str, Enum):
    """Status of a tool execution result."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class MCPToolResult:
    """Result from a tool execution."""

    tool_name: str
    status: MCPToolResultStatus
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == MCPToolResultStatus.SUCCESS

    def to_message_content(self) -> str:
        """Convert to a string suitable for message content."""
        if self.is_success:
            if isinstance(self.result, str):
                return self.result
            return str(self.result)
        return f"Error: {self.error}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


class MCPCapabilities(BaseModel):
    """Capabilities advertised by the tool registry."""

    tools: bool = True
    resources: bool = False
    prompts: bool = False
    logging: bool = True
    experimental: dict[str, bool] = Field(default_factory=dict)


class MCPServerInfo(BaseModel):
    """Information about the MCP server/registry."""

    name: str = "agent-engine-tools"
    version: str = "1.0.0"
    capabilities: MCPCapabilities = Field(default_factory=MCPCapabilities)
