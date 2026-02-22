"""SQLAlchemy models for persistence."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from agent_engine.agents.state import SubtaskStatus, TaskStatus


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Task(Base):
    """Task model for storing task metadata."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_request: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        default=TaskStatus.PENDING.value,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metrics
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)

    # Configuration
    max_tokens: Mapped[int] = mapped_column(Integer, default=100000)
    max_steps: Mapped[int] = mapped_column(Integer, default=50)
    max_tool_calls: Mapped[int] = mapped_column(Integer, default=100)
    max_iterations: Mapped[int] = mapped_column(Integer, default=10)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    subtasks: Mapped[list["TaskStep"]] = relationship(
        "TaskStep",
        back_populates="task",
        cascade="all, delete-orphan",
    )
    tool_calls: Mapped[list["ToolCallLog"]] = relationship(
        "ToolCallLog",
        back_populates="task",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_tasks_created_at", "created_at"),
        Index("ix_tasks_status_created", "status", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_request": self.user_request,
            "status": self.status,
            "error": self.error,
            "metrics": {
                "total_tokens": self.total_tokens,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "step_count": self.step_count,
                "tool_call_count": self.tool_call_count,
                "iteration_count": self.iteration_count,
            },
            "limits": {
                "max_tokens": self.max_tokens,
                "max_steps": self.max_steps,
                "max_tool_calls": self.max_tool_calls,
                "max_iterations": self.max_iterations,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TaskStep(Base):
    """Task step (subtask) model."""

    __tablename__ = "task_steps"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        default=SubtaskStatus.PENDING.value,
    )
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    dependencies: Mapped[dict] = mapped_column(JSON, default=list)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    task: Mapped["Task"] = relationship("Task", back_populates="subtasks")

    __table_args__ = (
        Index("ix_task_steps_task_sequence", "task_id", "sequence"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "sequence": self.sequence,
            "dependencies": self.dependencies,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ToolCallLog(Base):
    """Log of tool calls for auditing and debugging."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        index=True,
    )
    step_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    task: Mapped["Task"] = relationship("Task", back_populates="tool_calls")

    __table_args__ = (
        Index("ix_tool_calls_task_created", "task_id", "created_at"),
        Index("ix_tool_calls_tool_name", "tool_name"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Checkpoint(Base):
    """LangGraph checkpoint storage."""

    __tablename__ = "checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    checkpoint_id: Mapped[str] = mapped_column(String(64), index=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    checkpoint_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_checkpoints_thread_checkpoint", "thread_id", "checkpoint_id", unique=True),
    )
