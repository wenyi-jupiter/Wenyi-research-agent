"""Data access layer for persistence."""

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agent_engine.agents.state import SubtaskStatus, TaskStatus
from agent_engine.config import get_settings
from agent_engine.persistence.models import Task, TaskStep, ToolCallLog


class TaskRepository:
    """Repository for task operations."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession] | None = None,
    ):
        """Initialize the repository.

        Args:
            session_maker: Optional pre-configured session maker.
        """
        if session_maker:
            self._session_maker = session_maker
        else:
            settings = get_settings()
            engine = create_async_engine(settings.database_url, echo=False)
            self._session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

    async def create_task(
        self,
        task_id: str,
        user_request: str,
        max_tokens: int = 100000,
        max_steps: int = 50,
        max_tool_calls: int = 100,
        max_iterations: int = 10,
    ) -> Task:
        """Create a new task.

        Args:
            task_id: Unique task identifier.
            user_request: The user's request.
            max_tokens: Maximum token budget.
            max_steps: Maximum execution steps.
            max_tool_calls: Maximum tool calls.
            max_iterations: Maximum planning iterations.

        Returns:
            Created Task instance.
        """
        async with self._session_maker() as session:
            task = Task(
                id=task_id,
                user_request=user_request,
                status=TaskStatus.PENDING.value,
                max_tokens=max_tokens,
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
                max_iterations=max_iterations,
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task

    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task ID.

        Returns:
            Task instance or None if not found.
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(Task).where(Task.id == task_id)
            )
            return result.scalar_one_or_none()

    async def update_task(
        self,
        task_id: str,
        **updates: Any,
    ) -> Task | None:
        """Update a task.

        Args:
            task_id: The task ID.
            **updates: Fields to update.

        Returns:
            Updated Task instance or None.
        """
        async with self._session_maker() as session:
            await session.execute(
                update(Task)
                .where(Task.id == task_id)
                .values(**updates, updated_at=datetime.utcnow())
            )
            await session.commit()

            result = await session.execute(
                select(Task).where(Task.id == task_id)
            )
            return result.scalar_one_or_none()

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: str | None = None,
    ) -> Task | None:
        """Update task status.

        Args:
            task_id: The task ID.
            status: New status.
            error: Optional error message.

        Returns:
            Updated Task instance.
        """
        updates: dict[str, Any] = {"status": status.value}
        if error:
            updates["error"] = error
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            updates["completed_at"] = datetime.utcnow()

        return await self.update_task(task_id, **updates)

    async def update_task_metrics(
        self,
        task_id: str,
        total_tokens: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        step_count: int | None = None,
        tool_call_count: int | None = None,
        iteration_count: int | None = None,
    ) -> Task | None:
        """Update task metrics.

        Args:
            task_id: The task ID.
            total_tokens: Total token count.
            input_tokens: Input token count.
            output_tokens: Output token count.
            step_count: Step count.
            tool_call_count: Tool call count.
            iteration_count: Iteration count.

        Returns:
            Updated Task instance.
        """
        updates = {}
        if total_tokens is not None:
            updates["total_tokens"] = total_tokens
        if input_tokens is not None:
            updates["input_tokens"] = input_tokens
        if output_tokens is not None:
            updates["output_tokens"] = output_tokens
        if step_count is not None:
            updates["step_count"] = step_count
        if tool_call_count is not None:
            updates["tool_call_count"] = tool_call_count
        if iteration_count is not None:
            updates["iteration_count"] = iteration_count

        if updates:
            return await self.update_task(task_id, **updates)
        return await self.get_task(task_id)

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task and all related data.

        Args:
            task_id: The task ID.

        Returns:
            True if deleted, False if not found.
        """
        async with self._session_maker() as session:
            result = await session.execute(
                delete(Task).where(Task.id == task_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Task]:
        """List tasks with optional filtering.

        Args:
            status: Filter by status.
            limit: Maximum results.
            offset: Result offset.

        Returns:
            List of Task instances.
        """
        async with self._session_maker() as session:
            query = select(Task).order_by(Task.created_at.desc())

            if status:
                query = query.where(Task.status == status.value)

            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            return list(result.scalars().all())

    # Task Step operations

    async def create_task_step(
        self,
        step_id: str,
        task_id: str,
        description: str,
        sequence: int = 0,
        dependencies: list[str] | None = None,
    ) -> TaskStep:
        """Create a task step.

        Args:
            step_id: Unique step identifier.
            task_id: Parent task ID.
            description: Step description.
            sequence: Execution sequence.
            dependencies: List of dependency step IDs.

        Returns:
            Created TaskStep instance.
        """
        async with self._session_maker() as session:
            step = TaskStep(
                id=step_id,
                task_id=task_id,
                description=description,
                sequence=sequence,
                dependencies=dependencies or [],
            )
            session.add(step)
            await session.commit()
            await session.refresh(step)
            return step

    async def update_task_step(
        self,
        step_id: str,
        status: SubtaskStatus | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> TaskStep | None:
        """Update a task step.

        Args:
            step_id: The step ID.
            status: New status.
            result: Execution result.
            error: Error message.

        Returns:
            Updated TaskStep instance.
        """
        async with self._session_maker() as session:
            updates: dict[str, Any] = {}
            if status:
                updates["status"] = status.value
            if result is not None:
                updates["result"] = result
            if error is not None:
                updates["error"] = error
            if status in (SubtaskStatus.COMPLETED, SubtaskStatus.FAILED):
                updates["completed_at"] = datetime.utcnow()

            if updates:
                await session.execute(
                    update(TaskStep)
                    .where(TaskStep.id == step_id)
                    .values(**updates)
                )
                await session.commit()

            result_row = await session.execute(
                select(TaskStep).where(TaskStep.id == step_id)
            )
            return result_row.scalar_one_or_none()

    async def get_task_steps(self, task_id: str) -> list[TaskStep]:
        """Get all steps for a task.

        Args:
            task_id: The task ID.

        Returns:
            List of TaskStep instances.
        """
        async with self._session_maker() as session:
            result = await session.execute(
                select(TaskStep)
                .where(TaskStep.task_id == task_id)
                .order_by(TaskStep.sequence)
            )
            return list(result.scalars().all())

    # Tool call logging

    async def log_tool_call(
        self,
        task_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: str | None = None,
        success: bool = False,
        execution_time_ms: float = 0.0,
        step_id: str | None = None,
        retry_count: int = 0,
    ) -> ToolCallLog:
        """Log a tool call.

        Args:
            task_id: The task ID.
            tool_name: Name of the tool called.
            arguments: Tool arguments.
            result: Tool result.
            error: Error message if failed.
            success: Whether call succeeded.
            execution_time_ms: Execution time.
            step_id: Optional step ID.
            retry_count: Number of retries.

        Returns:
            Created ToolCallLog instance.
        """
        async with self._session_maker() as session:
            log = ToolCallLog(
                task_id=task_id,
                step_id=step_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                error=error,
                success=success,
                execution_time_ms=execution_time_ms,
                retry_count=retry_count,
            )
            session.add(log)
            await session.commit()
            await session.refresh(log)
            return log

    async def get_tool_calls(
        self,
        task_id: str,
        tool_name: str | None = None,
        limit: int = 100,
    ) -> list[ToolCallLog]:
        """Get tool calls for a task.

        Args:
            task_id: The task ID.
            tool_name: Optional filter by tool name.
            limit: Maximum results.

        Returns:
            List of ToolCallLog instances.
        """
        async with self._session_maker() as session:
            query = (
                select(ToolCallLog)
                .where(ToolCallLog.task_id == task_id)
                .order_by(ToolCallLog.created_at.desc())
            )

            if tool_name:
                query = query.where(ToolCallLog.tool_name == tool_name)

            query = query.limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())


# Global repository instance
_repository: TaskRepository | None = None


def get_task_repository() -> TaskRepository:
    """Get the global task repository instance."""
    global _repository
    if _repository is None:
        _repository = TaskRepository()
    return _repository
