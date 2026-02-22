"""WebSocket handling for real-time task streaming."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent_engine.agents import AgentOrchestrator
from agent_engine.memory import get_long_term_memory
from agent_engine.persistence import PostgresCheckpointSaver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str) -> None:
        """Accept and register a connection."""
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)

    def disconnect(self, websocket: WebSocket, task_id: str) -> None:
        """Remove a connection."""
        if task_id in self.active_connections:
            self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def send_to_task(self, task_id: str, message: dict) -> None:
        """Send message to all connections for a task."""
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

    async def broadcast(self, message: dict) -> None:
        """Broadcast to all connections."""
        for connections in self.active_connections.values():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

    async def close_all(self) -> None:
        """Close all active WebSocket connections gracefully.

        Sends a shutdown notification to each client before closing.
        """
        from datetime import datetime

        total = sum(len(conns) for conns in self.active_connections.values())
        if total == 0:
            return

        logger.info(f"Closing {total} active WebSocket connection(s)...")

        for task_id, connections in list(self.active_connections.items()):
            for ws in list(connections):
                try:
                    await ws.send_json({
                        "type": "server_shutdown",
                        "message": "Server is shutting down",
                        "timestamp": datetime.now().isoformat(),
                    })
                    await ws.close(code=1001, reason="Server shutdown")
                except Exception:
                    pass  # Connection may already be closed

        self.active_connections.clear()

    @property
    def connection_count(self) -> int:
        """Return total number of active connections."""
        return sum(len(conns) for conns in self.active_connections.values())


manager = ConnectionManager()

# Track running tasks to prevent double-starts
_running_tasks: set[str] = set()


def _save_task_result(task_id: str, last_state: dict | None, user_request: str) -> None:
    """Save the task result JSON and final report markdown to llm_logs/."""
    from agent_engine.llm_logger import LOG_DIR
    import json as _json
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(exist_ok=True)

    if last_state:
        result_file = LOG_DIR / f"task_result_{ts}_{task_id[:12]}.json"
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                _json.dump(last_state, f, ensure_ascii=False, indent=2, default=str)
            logger.info("[WS] Task result saved to %s", result_file)
        except Exception as e:
            logger.warning("[WS] Could not save task result: %s", e)

        final_report = last_state.get("final_report", "")
        if final_report:
            report_file = LOG_DIR / f"report_{ts}_{task_id[:12]}.md"
            try:
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(final_report)
                logger.info("[WS] Report saved to %s", report_file)
            except Exception as e:
                logger.warning("[WS] Could not save report: %s", e)


@router.websocket("/tasks/{task_id}/stream")
async def task_stream(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for streaming task execution.

    Message types sent to client:
    - connected: Connection established
    - status: Task status update
    - step: Step execution update
    - tool_call: Tool call started/completed
    - message: Agent message
    - error: Error occurred
    - completed: Task completed

    Client can send:
    - start: Start task execution with {"request": "user request"}
    - resume: Resume interrupted task
    - cancel: Cancel task
    """
    logger.info(f"[WS] New connection for task {task_id}")
    await manager.connect(websocket, task_id)

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"[WS] Connection confirmed for {task_id}")

        # Wait for client commands
        while True:
            data = await websocket.receive_json()
            command = data.get("command", "")
            logger.info(f"[WS] Received command: {command}")

            if command == "start":
                await _handle_start(websocket, task_id, data)

            elif command == "resume":
                await _handle_resume(websocket, task_id)

            elif command == "cancel":
                await _handle_cancel(websocket, task_id)

            elif command == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown command: {command}",
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })
        manager.disconnect(websocket, task_id)


async def _handle_start(
    websocket: WebSocket,
    task_id: str,
    data: dict[str, Any],
) -> None:
    """Handle start command."""
    logger.info(f"[WS] Starting task {task_id}")

    # Guard: prevent double-starting the same task
    if task_id in _running_tasks:
        logger.warning(f"[WS] Task {task_id} already running — ignoring duplicate start")
        await websocket.send_json({
            "type": "error",
            "message": "Task is already running",
        })
        return
    _running_tasks.add(task_id)

    # Clear stale dedup state for this task
    _msg_seq.pop(task_id, None)
    _last_step_count.pop(task_id, None)
    
    user_request = data.get("request", "")
    if not user_request:
        logger.error("[WS] Missing request field")
        _running_tasks.discard(task_id)
        await websocket.send_json({
            "type": "error",
            "message": "Missing 'request' field",
        })
        return

    logger.info(f"[WS] User request: {user_request[:100]}...")

    # Start LLM interaction logging for this task
    from agent_engine.llm_logger import begin_task_logging
    begin_task_logging(task_id, user_request)

    # Get memory context
    memory_context = data.get("context", "")
    if not memory_context:
        try:
            memory = get_long_term_memory()
            memory_context = await memory.get_context(user_request, limit=3)
        except Exception as e:
            logger.warning(f"[WS] Memory context error: {e}")
            memory_context = ""

    # Create orchestrator
    try:
        checkpointer = PostgresCheckpointSaver()
        logger.info("[WS] Checkpointer created")
    except Exception as e:
        logger.warning(f"[WS] Checkpointer error: {e}")
        checkpointer = None

    orchestrator = AgentOrchestrator(checkpointer=checkpointer)
    logger.info("[WS] Orchestrator created")

    await websocket.send_json({
        "type": "status",
        "task_id": task_id,
        "status": "starting",
        "message": "Task execution starting",
        "timestamp": datetime.now().isoformat(),
    })

    # Stream execution
    try:
        logger.info("[WS] Starting stream...")
        event_count = 0
        last_state = None
        
        async for event in orchestrator.stream(
            user_request=user_request,
            task_id=task_id,
            memory_context=memory_context,
        ):
            event_count += 1
            logger.info(f"[WS] Event {event_count}: {list(event.keys())}")
            await _send_event(websocket, task_id, event)
            
            # Keep track of last state (should contain final report)
            for node_name, node_state in event.items():
                if isinstance(node_state, dict):
                    last_state = node_state

        logger.info(f"[WS] Task completed with {event_count} events")
        
        # Build completed message with final state
        completed_msg = {
            "type": "completed",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Include final report, subtasks, citations, and metrics
        if last_state:
            final_report = last_state.get("final_report", "")
            if final_report:
                completed_msg["final_report"] = final_report

            messages_list = last_state.get("messages", [])
            if messages_list:
                for msg in reversed(messages_list):
                    content = None
                    if isinstance(msg, dict):
                        content = msg.get("content")
                    else:
                        content = getattr(msg, "content", None)
                    if isinstance(content, str) and content.strip():
                        if "Final report generated" in content or "report" in content.lower():
                            completed_msg["report_process_message"] = content
                            break
            
            subtasks = last_state.get("subtasks", [])
            if subtasks:
                completed_msg["subtasks"] = subtasks
            
            citations = last_state.get("citations", [])
            if citations:
                completed_msg["citations"] = citations
            
            tool_call_log = last_state.get("tool_call_log", [])
            if tool_call_log:
                completed_msg["tool_call_log"] = tool_call_log
            
            metrics = last_state.get("metrics", {})
            if metrics:
                completed_msg["step_count"] = metrics.get("step_count", 0)
                completed_msg["total_tokens"] = metrics.get("total_tokens", 0)
                completed_msg["tool_call_count"] = metrics.get("tool_call_count", 0)
        
        await websocket.send_json(completed_msg)

        # Save task result to llm_logs/
        _save_task_result(task_id, last_state, user_request)

    except Exception as e:
        logger.error(f"[WS] Stream error: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "task_id": task_id,
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        })
    finally:
        from agent_engine.llm_logger import finish_task_logging
        finish_task_logging()
        _running_tasks.discard(task_id)


async def _handle_resume(websocket: WebSocket, task_id: str) -> None:
    """Handle resume command."""
    try:
        checkpointer = PostgresCheckpointSaver()
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Checkpointer not available: {e}",
        })
        return

    orchestrator = AgentOrchestrator(checkpointer=checkpointer)

    await websocket.send_json({
        "type": "status",
        "task_id": task_id,
        "status": "resuming",
        "message": "Resuming task execution",
        "timestamp": datetime.now().isoformat(),
    })

    try:
        # Get current state and stream from there
        config = {"configurable": {"thread_id": task_id}}

        async for event in orchestrator.graph.astream(None, config=config):
            await _send_event(websocket, task_id, event)

        await websocket.send_json({
            "type": "completed",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "task_id": task_id,
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        })


async def _handle_cancel(websocket: WebSocket, task_id: str) -> None:
    """Handle cancel command."""
    from agent_engine.agents.state import TaskStatus
    from agent_engine.persistence import get_task_repository

    repo = get_task_repository()
    await repo.update_task_status(task_id, TaskStatus.CANCELLED)

    await websocket.send_json({
        "type": "cancelled",
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
    })


# Global message sequence counter per task for deduplication debugging
_msg_seq: dict[str, int] = {}

# ── Server-side deduplication (step_count based) ──
# LangGraph with PostgresCheckpointSaver emits TWO events per node execution:
# a "before" event (state entering the node, OLD step_count) and an "after" event
# (state after node returns, NEW step_count).  Each node increments step_count,
# so "before" events have the SAME step_count as the previous node's "after" event.
# By tracking the last sent step_count, we reliably filter out "before" events.
_last_step_count: dict[str, int] = {}


async def _send_event(
    websocket: WebSocket,
    task_id: str,
    event: dict[str, Any],
) -> None:
    """Send a graph event to the WebSocket."""
    # Extract node name and state from event
    for node_name, node_state in event.items():
        # Skip LangGraph internal pseudo-nodes
        if node_name.startswith("__"):
            continue

        # ── Server-side dedup: skip "before" events using step_count ──
        # Each node increments metrics.step_count.  "Before" events have the OLD
        # step_count (same as what we last sent), so step_count <= last → skip.
        if isinstance(node_state, dict):
            metrics_for_dedup = node_state.get("metrics", {})
            current_step = metrics_for_dedup.get("step_count", 0)
            last_step = _last_step_count.get(task_id, 0)
            if current_step <= last_step:
                logger.info(
                    f"[WS] DEDUP skip {node_name} step={current_step} <= last={last_step}"
                )
                continue
            _last_step_count[task_id] = current_step

        message_type = "step"

        # Determine message type based on node
        if node_name == "planner":
            message_type = "planning"
        elif node_name == "executor":
            message_type = "executing"
        elif node_name == "critic":
            message_type = "reviewing"
        elif node_name == "reporter":
            message_type = "report"

        # Build message
        message = {
            "type": message_type,
            "task_id": task_id,
            "node": node_name,
            "timestamp": datetime.now().isoformat(),
        }

        # Add relevant state info
        if isinstance(node_state, dict):
            message["status"] = node_state.get("status", "")
            message["iteration"] = node_state.get("iteration_count", 0)
            metrics = node_state.get("metrics", {})
            message["step_count"] = metrics.get("step_count", 0)
            message["total_tokens"] = metrics.get("total_tokens", 0)
            message["tool_call_count"] = metrics.get("tool_call_count", 0)

            # Add subtask info if available
            subtasks = node_state.get("subtasks", [])
            if subtasks:
                current_idx = node_state.get("current_subtask_index", 0)
                if node_name == "executor" and current_idx > 0:
                    # Executor already incremented the index after completing a subtask,
                    # so current_idx points to the NEXT subtask.  Show the one just completed.
                    message["current_subtask"] = subtasks[current_idx - 1]
                elif current_idx < len(subtasks):
                    message["current_subtask"] = subtasks[current_idx]
                # Always include full subtasks list so frontend can track progress
                message["subtasks"] = subtasks

            # Add tool call info if in executor results
            results = node_state.get("execution_results", [])
            if results:
                latest = results[-1]
                if "tool_results" in latest:
                    message["tool_calls"] = latest["tool_results"]
                # Include the completed subtask ID for accurate frontend display
                message["completed_subtask_id"] = latest.get("subtask_id", "")

            # Add tool call log (latest entries) for executor events
            tool_call_log = node_state.get("tool_call_log", [])
            if tool_call_log:
                message["tool_call_log"] = tool_call_log

            # Add citations for executor events
            citations = node_state.get("citations", [])
            if citations:
                message["citations"] = citations

            # Add critic feedback for critic events
            if node_name == "critic":
                critic_feedback = node_state.get("critic_feedback")
                if critic_feedback:
                    message["feedback"] = critic_feedback
                # Also include the evaluation message content if available
                messages_list = node_state.get("messages", [])
                if messages_list:
                    for msg in messages_list:
                        if hasattr(msg, "content") and "Critic evaluation" in str(msg.content):
                            message["evaluation_message"] = str(msg.content)
                            break

            # Add final report for reporter events
            final_report = node_state.get("final_report", "")
            if final_report:
                message["final_report"] = final_report

            # Add reporter process summary message (if available)
            if node_name == "reporter":
                messages_list = node_state.get("messages", [])
                if messages_list:
                    for msg in reversed(messages_list):
                        content = None
                        if isinstance(msg, dict):
                            content = msg.get("content")
                        else:
                            content = getattr(msg, "content", None)
                        if isinstance(content, str) and content.strip():
                            if "Final report generated" in content or "report" in content.lower():
                                message["report_process_message"] = content
                                break

        # Add sequence number for debugging
        if task_id not in _msg_seq:
            _msg_seq[task_id] = 0
        _msg_seq[task_id] += 1
        message["seq"] = _msg_seq[task_id]

        logger.info(
            f"[WS] SEND seq={_msg_seq[task_id]} type={message_type} "
            f"node={node_name} task={task_id}"
        )
        await websocket.send_json(message)
