"""Tests for agent state."""

import pytest
from datetime import datetime

from agent_engine.agents.state import (
    AgentState,
    CriticFeedback,
    ExecutionMetrics,
    GraphState,
    Subtask,
    SubtaskStatus,
    TaskStatus,
)


class TestSubtask:
    """Tests for Subtask dataclass."""

    def test_subtask_creation(self):
        """Test creating a subtask."""
        subtask = Subtask(
            id="subtask_001",
            description="Test subtask",
            dependencies=["subtask_000"],
        )

        assert subtask.id == "subtask_001"
        assert subtask.description == "Test subtask"
        assert subtask.status == SubtaskStatus.PENDING
        assert subtask.dependencies == ["subtask_000"]
        assert subtask.result is None

    def test_subtask_to_dict(self):
        """Test subtask serialization."""
        subtask = Subtask(
            id="subtask_001",
            description="Test",
            status=SubtaskStatus.COMPLETED,
            result={"key": "value"},
        )

        d = subtask.to_dict()

        assert d["id"] == "subtask_001"
        assert d["status"] == "completed"
        assert d["result"] == {"key": "value"}


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_add_tokens(self):
        """Test adding token counts."""
        metrics = ExecutionMetrics()
        metrics.add_tokens(input_tokens=100, output_tokens=50)

        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.total_tokens == 150

        metrics.add_tokens(input_tokens=50, output_tokens=25)
        assert metrics.total_tokens == 225

    def test_increment_step(self):
        """Test incrementing step counter."""
        metrics = ExecutionMetrics()
        assert metrics.step_count == 0

        metrics.increment_step()
        assert metrics.step_count == 1

        metrics.increment_step()
        assert metrics.step_count == 2

    def test_increment_tool_calls(self):
        """Test incrementing tool call counter."""
        metrics = ExecutionMetrics()
        metrics.increment_tool_calls(3)

        assert metrics.tool_call_count == 3


class TestCriticFeedback:
    """Tests for CriticFeedback dataclass."""

    def test_feedback_creation(self):
        """Test creating critic feedback."""
        feedback = CriticFeedback(
            is_complete=True,
            is_correct=True,
            feedback="Task completed successfully",
            confidence=0.95,
        )

        assert feedback.is_complete
        assert feedback.is_correct
        assert feedback.confidence == 0.95
        assert not feedback.needs_revision

    def test_feedback_needs_revision(self):
        """Test feedback indicating revision needed."""
        feedback = CriticFeedback(
            is_complete=False,
            is_correct=False,
            feedback="Task incomplete",
            suggestions=["Add error handling", "Update tests"],
            needs_revision=True,
        )

        assert feedback.needs_revision
        assert len(feedback.suggestions) == 2


class TestAgentState:
    """Tests for AgentState class."""

    def test_state_creation(self):
        """Test creating agent state."""
        state = AgentState(
            task_id="task_123",
            user_request="Test request",
        )

        assert state.task_id == "task_123"
        assert state.user_request == "Test request"
        assert state.status == TaskStatus.PENDING
        assert state.subtasks == []

    def test_state_to_dict(self):
        """Test state serialization."""
        state = AgentState(
            task_id="task_123",
            user_request="Test",
            status=TaskStatus.EXECUTING,
        )
        state.subtasks.append(
            Subtask(id="st_1", description="Subtask 1")
        )

        d = state.to_dict()

        assert d["task_id"] == "task_123"
        assert d["status"] == "executing"
        assert len(d["subtasks"]) == 1

    def test_state_from_dict(self):
        """Test state deserialization."""
        data = {
            "task_id": "task_456",
            "user_request": "From dict",
            "status": "completed",
            "messages": [],
            "subtasks": [
                {
                    "id": "st_1",
                    "description": "Deserialized subtask",
                    "status": "completed",
                }
            ],
            "metrics": {
                "total_tokens": 1000,
                "step_count": 5,
            },
        }

        state = AgentState.from_dict(data)

        assert state.task_id == "task_456"
        assert state.status == TaskStatus.COMPLETED
        assert len(state.subtasks) == 1
        assert state.metrics.total_tokens == 1000


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.PLANNING.value == "planning"
        assert TaskStatus.EXECUTING.value == "executing"
        assert TaskStatus.REVIEWING.value == "reviewing"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
