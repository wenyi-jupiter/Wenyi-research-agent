"""Agent state definitions for LangGraph workflow."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TaskStatus(str, Enum):
    """Status of a task in the workflow."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubtaskStatus(str, Enum):
    """Status of a subtask."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A subtask created by the planner."""

    id: str
    description: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "tool_calls": self.tool_calls,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ExecutionMetrics:
    """Metrics tracking for execution."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    step_count: int = 0
    tool_call_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens

    def increment_step(self) -> None:
        """Increment step counter."""
        self.step_count += 1

    def increment_tool_calls(self, count: int = 1) -> None:
        """Increment tool call counter."""
        self.tool_call_count += count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "step_count": self.step_count,
            "tool_call_count": self.tool_call_count,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class CriticFeedback:
    """Feedback from the critic agent."""

    is_complete: bool
    is_correct: bool
    feedback: str
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_revision: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_complete": self.is_complete,
            "is_correct": self.is_correct,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "needs_revision": self.needs_revision,
        }


class AgentState:
    """State container for the multi-agent workflow.

    This is a TypedDict-compatible class that LangGraph uses to track
    state across the workflow execution.
    """

    def __init__(
        self,
        task_id: str = "",
        user_request: str = "",
        messages: list[BaseMessage] | None = None,
        status: TaskStatus = TaskStatus.PENDING,
        subtasks: list[Subtask] | None = None,
        current_subtask_index: int = 0,
        execution_results: list[dict[str, Any]] | None = None,
        critic_feedback: CriticFeedback | None = None,
        metrics: ExecutionMetrics | None = None,
        memory_context: str = "",
        error: str | None = None,
        iteration_count: int = 0,
        max_iterations: int = 10,
    ):
        self.task_id = task_id
        self.user_request = user_request
        self.messages = messages or []
        self.status = status
        self.subtasks = subtasks or []
        self.current_subtask_index = current_subtask_index
        self.execution_results = execution_results or []
        self.critic_feedback = critic_feedback
        self.metrics = metrics or ExecutionMetrics()
        self.memory_context = memory_context
        self.error = error
        self.iteration_count = iteration_count
        self.max_iterations = max_iterations

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "user_request": self.user_request,
            "messages": [m.dict() for m in self.messages],
            "status": self.status.value,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "current_subtask_index": self.current_subtask_index,
            "execution_results": self.execution_results,
            "critic_feedback": self.critic_feedback.to_dict() if self.critic_feedback else None,
            "metrics": self.metrics.to_dict(),
            "memory_context": self.memory_context,
            "error": self.error,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Create state from dictionary."""
        from langchain_core.messages import messages_from_dict

        subtasks = [
            Subtask(
                id=s["id"],
                description=s["description"],
                status=SubtaskStatus(s["status"]),
                dependencies=s.get("dependencies", []),
                tool_calls=s.get("tool_calls", []),
                result=s.get("result"),
                error=s.get("error"),
            )
            for s in data.get("subtasks", [])
        ]

        critic_feedback = None
        if data.get("critic_feedback"):
            cf = data["critic_feedback"]
            critic_feedback = CriticFeedback(
                is_complete=cf["is_complete"],
                is_correct=cf["is_correct"],
                feedback=cf["feedback"],
                suggestions=cf.get("suggestions", []),
                confidence=cf.get("confidence", 0.0),
                needs_revision=cf.get("needs_revision", False),
            )

        metrics = ExecutionMetrics()
        if data.get("metrics"):
            m = data["metrics"]
            metrics.total_tokens = m.get("total_tokens", 0)
            metrics.input_tokens = m.get("input_tokens", 0)
            metrics.output_tokens = m.get("output_tokens", 0)
            metrics.step_count = m.get("step_count", 0)
            metrics.tool_call_count = m.get("tool_call_count", 0)

        return cls(
            task_id=data.get("task_id", ""),
            user_request=data.get("user_request", ""),
            messages=messages_from_dict(data.get("messages", [])),
            status=TaskStatus(data.get("status", "pending")),
            subtasks=subtasks,
            current_subtask_index=data.get("current_subtask_index", 0),
            execution_results=data.get("execution_results", []),
            critic_feedback=critic_feedback,
            metrics=metrics,
            memory_context=data.get("memory_context", ""),
            error=data.get("error"),
            iteration_count=data.get("iteration_count", 0),
            max_iterations=data.get("max_iterations", 10),
        )


# TypedDict for LangGraph compatibility
from typing import TypedDict


class GraphState(TypedDict, total=False):
    """TypedDict state for LangGraph graph."""

    task_id: str
    user_request: str
    messages: Annotated[list[BaseMessage], add_messages]
    status: str
    subtasks: list[dict[str, Any]]
    current_subtask_index: int
    execution_results: list[dict[str, Any]]
    critic_feedback: dict[str, Any] | None
    metrics: dict[str, Any]
    memory_context: str
    error: str | None
    iteration_count: int
    max_iterations: int
    # Budget tracking
    total_tokens: int
    max_tokens: int
    max_steps: int
    max_tool_calls: int
    # Citation tracking
    citations: list[dict[str, Any]]  # [{id, title, url, snippet, source_tool, accessed_at}]
    # Final report with citations
    final_report: str
    # Detailed tool call log
    tool_call_log: list[dict[str, Any]]  # [{tool_name, args, result_summary, success, timestamp, subtask_id}]
    # P2/P3: Structured evidence claims from Validator node
    # Each entry: {subtask_id, claim, grounded, evidence, source_url, confidence}
    evidence_claims: list[dict[str, Any]]
    # P6: Inter-subtask evidence pool — fetch_url results shared across all subtasks.
    # Each entry: {url, content, keywords, subtask_id, quality_score}
    # Allows subtask N to borrow high-quality evidence from subtask M even when M
    # completed before N started (solving the "information silo" problem).
    global_evidence_pool: list[dict[str, Any]]
    # P7: Strategies that have been attempted and failed — injected into the Planner
    # at replan time so it does NOT retry the same approach.
    # Each entry is a short string describing the failed strategy.
    tried_strategies: list[str]
    # P8: Overall data quality level computed by the Validator.
    # "good" (grounding ≥ 70%), "partial" (40-70%), "poor" (< 40%).
    # Used by Reporter to decide how many disclaimers to add.
    data_quality_level: str
