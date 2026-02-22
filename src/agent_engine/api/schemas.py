"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============ Task Schemas ============


class TaskCreate(BaseModel):
    """Request to create a new task."""

    request: str = Field(..., description="The user's request")
    max_tokens: int = Field(default=100_000, description="Maximum token budget")
    max_steps: int = Field(default=50, description="Maximum execution steps")
    max_tool_calls: int = Field(default=100, description="Maximum tool calls")
    max_iterations: int = Field(default=10, description="Maximum planning iterations")
    context: str | None = Field(default=None, description="Optional context from memory")


class TaskMetrics(BaseModel):
    """Task execution metrics."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    step_count: int = 0
    tool_call_count: int = 0
    iteration_count: int = 0


class TaskLimits(BaseModel):
    """Task execution limits."""

    max_tokens: int = 100_000
    max_steps: int = 50
    max_tool_calls: int = 100
    max_iterations: int = 10


class SubtaskResponse(BaseModel):
    """Subtask information."""

    id: str
    description: str
    status: str
    dependencies: list[str] = []
    result: Any = None
    error: str | None = None


class CitationResponse(BaseModel):
    """Citation/source reference."""

    id: int
    title: str = ""
    url: str = ""
    snippet: str = ""
    source_tool: str = ""
    accessed_at: str | None = None
    verified: bool | None = None
    http_status: int | None = None


class ToolCallLogEntry(BaseModel):
    """Detailed tool call log entry."""

    tool_name: str
    args: dict[str, Any] = {}
    result_summary: str = ""
    success: bool = True
    timestamp: str | None = None
    subtask_id: str = ""
    execution_time_ms: float = 0.0


class TaskResponse(BaseModel):
    """Task response with full details."""

    id: str
    user_request: str
    status: str
    error: str | None = None
    metrics: TaskMetrics
    limits: TaskLimits
    subtasks: list[SubtaskResponse] = []
    citations: list[CitationResponse] = []
    final_report: str = ""
    tool_call_log: list[ToolCallLogEntry] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class TaskListResponse(BaseModel):
    """Response for task list endpoint."""

    tasks: list[TaskResponse]
    total: int
    limit: int
    offset: int


class TaskStatusResponse(BaseModel):
    """Brief task status response."""

    id: str
    status: str
    step_count: int
    total_tokens: int
    error: str | None = None


# ============ Tool Schemas ============


class ToolParameter(BaseModel):
    """Tool parameter schema."""

    type: str = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


class ToolSchema(BaseModel):
    """Tool schema for registration."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Tool description")
    parameters: ToolParameter = Field(default_factory=ToolParameter)
    tags: list[str] = []


class ToolResponse(BaseModel):
    """Tool information response."""

    name: str
    description: str
    parameters: ToolParameter
    tags: list[str] = []
    version: str = "1.0.0"


class ToolListResponse(BaseModel):
    """Response for tool list endpoint."""

    tools: list[ToolResponse]
    total: int


class ToolCallRequest(BaseModel):
    """Request to call a tool directly."""

    tool_name: str
    arguments: dict[str, Any] = {}


class ToolCallResponse(BaseModel):
    """Response from a tool call."""

    tool_name: str
    status: str
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


# ============ Memory Schemas ============


class MemoryStore(BaseModel):
    """Request to store memory."""

    content: str = Field(..., description="Content to store")
    content_type: str = Field(default="text", description="Type of content")
    task_id: str | None = None
    metadata: dict[str, Any] | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemorySearch(BaseModel):
    """Request to search memory."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, ge=1, le=20)
    content_type: str | None = None
    task_id: str | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class MemorySearchResult(BaseModel):
    """Memory search result."""

    id: int
    content: str
    content_type: str
    similarity: float
    task_id: str | None
    metadata: dict[str, Any] | None
    created_at: datetime


class MemorySearchResponse(BaseModel):
    """Response for memory search."""

    results: list[MemorySearchResult]
    query: str
    total: int


# ============ WebSocket Schemas ============


class WSMessage(BaseModel):
    """WebSocket message."""

    type: str = Field(..., description="Message type")
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class WSTaskUpdate(BaseModel):
    """WebSocket task update message."""

    task_id: str
    status: str
    current_step: int
    total_steps: int
    message: str
    subtask_id: str | None = None
    tool_name: str | None = None


# ============ Error Schemas ============


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
    code: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    database: str = "unknown"
    tools_count: int = 0
