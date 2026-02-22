"""FastAPI REST routes."""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from agent_engine.agents import AgentOrchestrator, TaskStatus
from agent_engine.api.schemas import (
    CitationResponse,
    ErrorResponse,
    HealthResponse,
    MemorySearch,
    MemorySearchResponse,
    MemorySearchResult,
    MemoryStore,
    TaskCreate,
    TaskLimits,
    TaskListResponse,
    TaskMetrics,
    TaskResponse,
    TaskStatusResponse,
    ToolCallLogEntry,
    ToolCallRequest,
    ToolCallResponse,
    ToolListResponse,
    ToolParameter,
    ToolResponse,
    ToolSchema,
)
from agent_engine.memory import get_long_term_memory
from agent_engine.persistence import PostgresCheckpointSaver, get_task_repository
from agent_engine.tools import MCPToolRequest, get_tool_executor, get_tool_registry

logger = logging.getLogger(__name__)

router = APIRouter()


# ============ Health ============


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    from agent_engine import __version__

    registry = get_tool_registry()

    # Actual database check
    db_status = "unknown"
    try:
        from agent_engine.main import get_shared_db_engine
        engine = get_shared_db_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return HealthResponse(
        status="ok",
        version=__version__,
        database=db_status,
        tools_count=len(registry),
    )


# ============ Process Management ============


@router.get(
    "/process",
    tags=["process"],
    summary="Get process information",
)
async def process_info() -> dict:
    """Get current server process information."""
    from agent_engine.api.websocket import manager
    from agent_engine.lifecycle import PID_FILE

    pid = os.getpid()
    pid_file_exists = PID_FILE.exists()
    pid_in_file = None
    if pid_file_exists:
        try:
            pid_in_file = int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pass

    return {
        "pid": pid,
        "pid_file": str(PID_FILE),
        "pid_file_exists": pid_file_exists,
        "pid_in_file": pid_in_file,
        "websocket_connections": manager.connection_count,
        "timestamp": datetime.now().isoformat(),
    }


@router.post(
    "/shutdown",
    tags=["process"],
    summary="Graceful server shutdown",
    responses={200: {"description": "Shutdown initiated"}},
)
async def shutdown() -> JSONResponse:
    """Initiate a graceful server shutdown.

    This endpoint:
    1. Notifies all WebSocket clients
    2. Triggers uvicorn shutdown via SIGINT
    3. The lifespan handler then cleans up DB connections and PID file

    Returns immediately with a confirmation; actual shutdown happens shortly after.
    """
    logger.info("Shutdown requested via API")

    async def _delayed_shutdown():
        """Delay shutdown briefly so the HTTP response can be sent first."""
        await asyncio.sleep(0.5)
        from agent_engine.lifecycle import request_shutdown
        request_shutdown()

    # Schedule shutdown in background (non-blocking so the response goes out)
    asyncio.create_task(_delayed_shutdown())

    return JSONResponse(
        status_code=200,
        content={
            "status": "shutdown_initiated",
            "message": "Server will shut down shortly",
            "pid": os.getpid(),
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============ Tasks ============


@router.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["tasks"],
    responses={400: {"model": ErrorResponse}},
)
async def create_task(request: TaskCreate) -> TaskResponse:
    """Create and execute a new task."""
    task_id = f"task_{uuid.uuid4().hex[:12]}"

    # Get memory context if not provided
    memory_context = request.context or ""
    if not memory_context:
        try:
            memory = get_long_term_memory()
            memory_context = await memory.get_context(request.request, limit=3)
        except Exception:
            memory_context = ""

    # Create orchestrator with checkpointing
    try:
        checkpointer = PostgresCheckpointSaver()
    except Exception:
        checkpointer = None

    orchestrator = AgentOrchestrator(checkpointer=checkpointer)

    # Run the task
    try:
        final_state = await orchestrator.run(
            user_request=request.request,
            task_id=task_id,
            memory_context=memory_context,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # Convert state to response
    return _state_to_response(task_id, request.request, final_state)


@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def get_task(task_id: str) -> TaskResponse:
    """Get task details by ID."""
    repo = get_task_repository()
    task = await repo.get_task(task_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    # Get subtasks
    steps = await repo.get_task_steps(task_id)

    return TaskResponse(
        id=task.id,
        user_request=task.user_request,
        status=task.status,
        error=task.error,
        metrics=TaskMetrics(
            total_tokens=task.total_tokens,
            input_tokens=task.input_tokens,
            output_tokens=task.output_tokens,
            step_count=task.step_count,
            tool_call_count=task.tool_call_count,
            iteration_count=task.iteration_count,
        ),
        limits=TaskLimits(
            max_tokens=task.max_tokens,
            max_steps=task.max_steps,
            max_tool_calls=task.max_tool_calls,
            max_iterations=task.max_iterations,
        ),
        subtasks=[
            {
                "id": s.id,
                "description": s.description,
                "status": s.status,
                "dependencies": s.dependencies,
                "result": s.result,
                "error": s.error,
            }
            for s in steps
        ],
        created_at=task.created_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
    )


@router.get(
    "/tasks/{task_id}/status",
    response_model=TaskStatusResponse,
    tags=["tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get brief task status."""
    repo = get_task_repository()
    task = await repo.get_task(task_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    return TaskStatusResponse(
        id=task.id,
        status=task.status,
        step_count=task.step_count,
        total_tokens=task.total_tokens,
        error=task.error,
    )


@router.post(
    "/tasks/{task_id}/resume",
    response_model=TaskResponse,
    tags=["tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def resume_task(task_id: str) -> TaskResponse:
    """Resume an interrupted task."""
    try:
        checkpointer = PostgresCheckpointSaver()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Checkpointer not available: {e}",
        )

    orchestrator = AgentOrchestrator(checkpointer=checkpointer)

    try:
        final_state = await orchestrator.resume(task_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    return _state_to_response(
        task_id,
        final_state.get("user_request", ""),
        final_state,
    )


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["tasks"],
    responses={404: {"model": ErrorResponse}},
)
async def cancel_task(task_id: str) -> None:
    """Cancel a running task."""
    repo = get_task_repository()
    task = await repo.get_task(task_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    await repo.update_task_status(task_id, TaskStatus.CANCELLED)


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    tags=["tasks"],
)
async def list_tasks(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TaskListResponse:
    """List tasks with optional filtering."""
    repo = get_task_repository()

    task_status = TaskStatus(status) if status else None
    tasks = await repo.list_tasks(status=task_status, limit=limit, offset=offset)

    return TaskListResponse(
        tasks=[
            TaskResponse(
                id=t.id,
                user_request=t.user_request,
                status=t.status,
                error=t.error,
                metrics=TaskMetrics(
                    total_tokens=t.total_tokens,
                    input_tokens=t.input_tokens,
                    output_tokens=t.output_tokens,
                    step_count=t.step_count,
                    tool_call_count=t.tool_call_count,
                    iteration_count=t.iteration_count,
                ),
                limits=TaskLimits(
                    max_tokens=t.max_tokens,
                    max_steps=t.max_steps,
                    max_tool_calls=t.max_tool_calls,
                    max_iterations=t.max_iterations,
                ),
                created_at=t.created_at,
                updated_at=t.updated_at,
                completed_at=t.completed_at,
            )
            for t in tasks
        ],
        total=len(tasks),
        limit=limit,
        offset=offset,
    )


# ============ Tools ============


@router.get(
    "/tools",
    response_model=ToolListResponse,
    tags=["tools"],
)
async def list_tools(
    tags: list[str] | None = Query(default=None, description="Filter by tags"),
) -> ToolListResponse:
    """List available tools."""
    registry = get_tool_registry()
    tools = registry.list_tools(tags=tags)

    return ToolListResponse(
        tools=[
            ToolResponse(
                name=t.name,
                description=t.description,
                parameters=ToolParameter(
                    type=t.parameters.type,
                    properties=t.parameters.properties,
                    required=t.parameters.required,
                ),
                tags=t.tags,
                version=t.version,
            )
            for t in tools
        ],
        total=len(tools),
    )


@router.post(
    "/tools",
    response_model=ToolResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["tools"],
    responses={400: {"model": ErrorResponse}},
)
async def register_tool(schema: ToolSchema) -> ToolResponse:
    """Register a new tool."""
    from agent_engine.tools.mcp_protocol import MCPParameterSchema, MCPToolSchema

    registry = get_tool_registry()

    if schema.name in registry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tool already exists: {schema.name}",
        )

    mcp_schema = MCPToolSchema(
        name=schema.name,
        description=schema.description,
        parameters=MCPParameterSchema(
            type=schema.parameters.type,
            properties=schema.parameters.properties,
            required=schema.parameters.required,
        ),
        tags=schema.tags,
    )

    # Register with a placeholder implementation
    async def placeholder_impl(**kwargs):
        return {"message": f"Tool {schema.name} called", "args": kwargs}

    registry.register_tool(mcp_schema, placeholder_impl)

    return ToolResponse(
        name=schema.name,
        description=schema.description,
        parameters=schema.parameters,
        tags=schema.tags,
    )


@router.post(
    "/tools/call",
    response_model=ToolCallResponse,
    tags=["tools"],
    responses={404: {"model": ErrorResponse}},
)
async def call_tool(request: ToolCallRequest) -> ToolCallResponse:
    """Call a tool directly."""
    registry = get_tool_registry()

    if request.tool_name not in registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {request.tool_name}",
        )

    executor = get_tool_executor()
    mcp_request = MCPToolRequest(
        tool_name=request.tool_name,
        arguments=request.arguments,
    )

    result = await executor.execute(mcp_request)

    return ToolCallResponse(
        tool_name=result.tool_name,
        status=result.status.value,
        result=result.result,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
    )


# ============ Memory ============


@router.post(
    "/memory",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    tags=["memory"],
)
async def store_memory(request: MemoryStore) -> dict:
    """Store content in long-term memory."""
    memory = get_long_term_memory()

    entry_id = await memory.store(
        content=request.content,
        content_type=request.content_type,
        task_id=request.task_id,
        metadata=request.metadata,
        importance=request.importance,
    )

    return {"id": entry_id, "status": "stored"}


@router.post(
    "/memory/search",
    response_model=MemorySearchResponse,
    tags=["memory"],
)
async def search_memory(request: MemorySearch) -> MemorySearchResponse:
    """Search long-term memory."""
    memory = get_long_term_memory()

    results = await memory.search(
        query=request.query,
        limit=request.limit,
        content_type=request.content_type,
        task_id=request.task_id,
        min_similarity=request.min_similarity,
    )

    return MemorySearchResponse(
        results=[
            MemorySearchResult(
                id=r.id,
                content=r.content,
                content_type=r.content_type,
                similarity=r.similarity,
                task_id=r.task_id,
                metadata=r.metadata,
                created_at=r.created_at,
            )
            for r in results
        ],
        query=request.query,
        total=len(results),
    )


# ============ Helpers ============


def _state_to_response(
    task_id: str,
    user_request: str,
    state: dict[str, Any],
) -> TaskResponse:
    """Convert graph state to response."""
    metrics = state.get("metrics", {})
    subtasks = state.get("subtasks", [])

    # Convert citations
    raw_citations = state.get("citations", [])
    citations = [
        CitationResponse(
            id=c.get("id", 0),
            title=c.get("title", ""),
            url=c.get("url", ""),
            snippet=c.get("snippet", ""),
            source_tool=c.get("source_tool", ""),
            accessed_at=c.get("accessed_at"),
            verified=c.get("verified"),
            http_status=c.get("http_status"),
        )
        for c in raw_citations
    ]

    # Convert tool call log
    raw_tool_log = state.get("tool_call_log", [])
    tool_call_log = [
        ToolCallLogEntry(
            tool_name=t.get("tool_name", ""),
            args=t.get("args", {}),
            result_summary=t.get("result_summary", ""),
            success=t.get("success", True),
            timestamp=t.get("timestamp"),
            subtask_id=t.get("subtask_id", ""),
            execution_time_ms=t.get("execution_time_ms", 0.0),
        )
        for t in raw_tool_log
    ]

    return TaskResponse(
        id=task_id,
        user_request=user_request,
        status=state.get("status", "unknown"),
        error=state.get("error"),
        metrics=TaskMetrics(
            total_tokens=metrics.get("total_tokens", 0),
            input_tokens=metrics.get("input_tokens", 0),
            output_tokens=metrics.get("output_tokens", 0),
            step_count=metrics.get("step_count", 0),
            tool_call_count=metrics.get("tool_call_count", 0),
            iteration_count=state.get("iteration_count", 0),
        ),
        limits=TaskLimits(
            max_tokens=state.get("max_tokens", 100_000),
            max_steps=state.get("max_steps", 50),
            max_tool_calls=state.get("max_tool_calls", 100),
            max_iterations=state.get("max_iterations", 10),
        ),
        subtasks=subtasks,
        citations=citations,
        final_report=state.get("final_report", ""),
        tool_call_log=tool_call_log,
    )
