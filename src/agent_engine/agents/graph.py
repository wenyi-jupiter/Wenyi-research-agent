"""Main LangGraph workflow for multi-agent collaboration."""

import uuid
from datetime import datetime
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from agent_engine.agents.critic import critic_node, route_after_critic
from agent_engine.agents.executor import executor_node, should_continue_executing
from agent_engine.agents.planner import planner_node
from agent_engine.agents.reporter import reporter_node
from agent_engine.agents.validator import validator_node
from agent_engine.agents.state import GraphState
from agent_engine.config import get_settings


def create_agent_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    """Create the multi-agent workflow graph.

    Args:
        checkpointer: Optional checkpoint saver for persistence.

    Returns:
        Compiled StateGraph.
    """
    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("validator", validator_node)   # P2: dedicated validator
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add edges
    workflow.add_edge("planner", "executor")
    # When executor finishes all subtasks, go directly to reporter per policy.
    workflow.add_conditional_edges(
        "executor",
        should_continue_executing,
        {
            "executor": "executor",
            # Policy: after subtask execution, skip validator/critic post checks.
            # Only enforce citation-summary consistency in reporter stage.
            "critic": "reporter",
            "end": END,
        },
    )
    # Validator always feeds into critic
    workflow.add_edge("validator", "critic")
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "planner": "planner",
            "reporter": "reporter",
            "end": END,
        },
    )
    # Reporter always goes to END
    workflow.add_edge("reporter", END)

    # Compile with checkpointer if provided
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


def create_initial_state(
    user_request: str,
    task_id: str | None = None,
    memory_context: str = "",
    max_iterations: int | None = None,
) -> GraphState:
    """Create initial state for a new task.

    Args:
        user_request: The user's request.
        task_id: Optional task ID (generated if not provided).
        memory_context: Context from long-term memory.
        max_iterations: Maximum planning iterations.

    Returns:
        Initial GraphState.
    """
    settings = get_settings()

    return {
        "task_id": task_id or f"task_{uuid.uuid4().hex[:12]}",
        "user_request": user_request,
        "messages": [],
        "status": "planning",
        "subtasks": [],
        "current_subtask_index": 0,
        "execution_results": [],
        "critic_feedback": None,
        "metrics": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "step_count": 0,
            "tool_call_count": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
        },
        "memory_context": memory_context,
        "error": None,
        "iteration_count": 0,
        "max_iterations": max_iterations or 10,
        "total_tokens": 0,
        "max_tokens": settings.max_tokens,
        "max_steps": settings.max_steps,
        "max_tool_calls": settings.max_tool_calls,
        "citations": [],
        "final_report": "",
        "tool_call_log": [],
        "evidence_claims": [],      # P2/P3: structured claim鈫抏vidence pairs
        "global_evidence_pool": [], # P6: cross-subtask shared fetch_url evidence
        "tried_strategies": [],     # P7: failed strategy strings for Planner blacklist
        "data_quality_level": "good",  # P8: "good" | "partial" | "poor"
        # Controls executor result reuse behavior across replans.
        # - "fuzzy": allow reuse based on ID or description similarity
        # - "strict": reuse ONLY when subtask fingerprint matches (i.e., same constraints)
        "reuse_mode": "fuzzy",
    }


class AgentOrchestrator:
    """Orchestrator for managing agent workflow execution."""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            checkpointer: Optional checkpoint saver.
        """
        self.checkpointer = checkpointer
        self.graph = create_agent_graph(checkpointer)

    async def run(
        self,
        user_request: str,
        task_id: str | None = None,
        memory_context: str = "",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the agent workflow for a user request.

        Args:
            user_request: The user's request.
            task_id: Optional task ID.
            memory_context: Context from memory.
            config: Optional LangGraph config.

        Returns:
            Final state as dictionary.
        """
        initial_state = create_initial_state(
            user_request=user_request,
            task_id=task_id,
            memory_context=memory_context,
        )

        # Configure thread for checkpointing
        run_config = config or {}
        if self.checkpointer and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": initial_state["task_id"]}

        # Run the graph
        # recursion_limit must be high enough to accommodate:
        # planner + N subtasks (each = executor node) + critic + possible replanning loop
        # A 8-subtask plan with 1 revision = ~25 nodes; set to 120 for safety.
        if "recursion_limit" not in run_config:
            run_config["recursion_limit"] = 120
        final_state = await self.graph.ainvoke(initial_state, config=run_config)

        # Update end time
        if "metrics" in final_state:
            final_state["metrics"]["end_time"] = datetime.now().isoformat()

        return final_state

    async def resume(
        self,
        task_id: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resume a previously interrupted task.

        Args:
            task_id: The task ID to resume.
            config: Optional LangGraph config.

        Returns:
            Final state as dictionary.

        Raises:
            ValueError: If task not found or checkpointer not configured.
        """
        if not self.checkpointer:
            raise ValueError("Checkpointer required for resume")

        run_config = config or {}
        run_config["configurable"] = {"thread_id": task_id}

        # Get current state
        state = await self.graph.aget_state(run_config)

        if not state or not state.values:
            raise ValueError(f"Task not found: {task_id}")

        # Resume execution
        final_state = await self.graph.ainvoke(None, config=run_config)

        return final_state

    async def stream(
        self,
        user_request: str,
        task_id: str | None = None,
        memory_context: str = "",
        config: dict[str, Any] | None = None,
    ):
        """Stream the agent workflow execution.

        Args:
            user_request: The user's request.
            task_id: Optional task ID.
            memory_context: Context from memory.
            config: Optional LangGraph config.

        Yields:
            State updates as they occur.
        """
        initial_state = create_initial_state(
            user_request=user_request,
            task_id=task_id,
            memory_context=memory_context,
        )

        run_config = config or {}
        if self.checkpointer and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": initial_state["task_id"]}

        async for event in self.graph.astream(
            initial_state, config=run_config, stream_mode="updates"
        ):
            yield event

    def get_state(self, task_id: str) -> dict[str, Any] | None:
        """Get current state of a task.

        Args:
            task_id: The task ID.

        Returns:
            Current state or None if not found.
        """
        if not self.checkpointer:
            return None

        config = {"configurable": {"thread_id": task_id}}

        try:
            state = self.graph.get_state(config)
            return state.values if state else None
        except Exception:
            return None
