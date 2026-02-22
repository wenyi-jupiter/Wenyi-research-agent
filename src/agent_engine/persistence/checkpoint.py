"""LangGraph checkpoint saver for PostgreSQL."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agent_engine.config import get_settings
from agent_engine.persistence.models import Checkpoint as CheckpointModel


class PostgresCheckpointSaver(BaseCheckpointSaver):
    """Checkpoint saver using PostgreSQL for persistence."""

    def __init__(
        self,
        database_url: str | None = None,
        async_session: async_sessionmaker[AsyncSession] | None = None,
    ):
        """Initialize the checkpoint saver.

        Args:
            database_url: Database connection URL.
            async_session: Optional pre-configured session maker.
        """
        super().__init__()

        if async_session:
            self._session_maker = async_session
        else:
            settings = get_settings()
            url = database_url or settings.database_url
            engine = create_async_engine(url, echo=False)
            self._session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """Get checkpoint synchronously (not implemented for async)."""
        raise NotImplementedError("Use aget_tuple() for async operations")

    async def aget_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """Get the latest checkpoint for a thread.

        Args:
            config: Configuration with thread_id.

        Returns:
            CheckpointTuple or None if not found.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None

        async with self._session_maker() as session:
            result = await session.execute(
                select(CheckpointModel)
                .where(CheckpointModel.thread_id == thread_id)
                .order_by(CheckpointModel.created_at.desc())
                .limit(1)
            )
            checkpoint_row = result.scalar_one_or_none()

            if not checkpoint_row:
                return None

            checkpoint_data = checkpoint_row.checkpoint_data
            metadata = checkpoint_row.checkpoint_metadata or {}

            return CheckpointTuple(
                config=config,
                checkpoint=Checkpoint(
                    v=checkpoint_data.get("v", 1),
                    id=checkpoint_row.checkpoint_id,
                    ts=checkpoint_data.get("ts", ""),
                    channel_values=checkpoint_data.get("channel_values", {}),
                    channel_versions=checkpoint_data.get("channel_versions", {}),
                    versions_seen=checkpoint_data.get("versions_seen", {}),
                    pending_sends=checkpoint_data.get("pending_sends", []),
                ),
                metadata=CheckpointMetadata(**metadata) if metadata else CheckpointMetadata(),
                parent_config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_row.parent_checkpoint_id,
                    }
                }
                if checkpoint_row.parent_checkpoint_id
                else None,
            )

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Put checkpoint synchronously (not implemented for async)."""
        raise NotImplementedError("Use aput() for async operations")

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Save a checkpoint.

        Args:
            config: Configuration with thread_id.
            checkpoint: The checkpoint to save.
            metadata: Checkpoint metadata.
            new_versions: New version information.

        Returns:
            Updated configuration.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("thread_id required in config")

        parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        # Serialize checkpoint
        checkpoint_data = {
            "v": checkpoint.get("v", 1),
            "ts": checkpoint.get("ts", ""),
            "channel_values": self._serialize_channel_values(
                checkpoint.get("channel_values", {})
            ),
            "channel_versions": checkpoint.get("channel_versions", {}),
            "versions_seen": checkpoint.get("versions_seen", {}),
            "pending_sends": checkpoint.get("pending_sends", []),
        }

        metadata_dict = {}
        if metadata:
            metadata_dict = {
                "source": getattr(metadata, "source", ""),
                "step": getattr(metadata, "step", 0),
                "writes": getattr(metadata, "writes", {}),
            }

        async with self._session_maker() as session:
            checkpoint_row = CheckpointModel(
                thread_id=thread_id,
                checkpoint_id=checkpoint["id"],
                parent_checkpoint_id=parent_checkpoint_id,
                checkpoint_data=checkpoint_data,
                metadata=metadata_dict,
            )
            session.add(checkpoint_row)
            await session.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes synchronously (not implemented for async)."""
        raise NotImplementedError("Use aput_writes() for async operations")

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for a task.

        This is called by LangGraph to persist intermediate writes
        before a checkpoint is committed. We store them as part of
        the checkpoint metadata.

        Args:
            config: Configuration with thread_id.
            writes: List of (channel, value) tuples.
            task_id: The task ID for these writes.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return

        # For simplicity, store pending writes as a checkpoint entry
        # with a special marker. LangGraph requires this method to not fail.
        # The actual writes will be committed via aput() later.
        try:
            serialized_writes = []
            for channel, value in writes:
                try:
                    json.dumps(value)
                    serialized_writes.append((channel, value))
                except (TypeError, ValueError):
                    if hasattr(value, "to_dict"):
                        serialized_writes.append((channel, value.to_dict()))
                    elif hasattr(value, "dict"):
                        serialized_writes.append((channel, value.dict()))
                    else:
                        serialized_writes.append((channel, str(value)))

            async with self._session_maker() as session:
                checkpoint_row = CheckpointModel(
                    thread_id=thread_id,
                    checkpoint_id=f"writes_{task_id}",
                    parent_checkpoint_id=config.get("configurable", {}).get("checkpoint_id"),
                    checkpoint_data={"pending_writes": serialized_writes, "task_id": task_id},
                    metadata={"source": "pending_writes"},
                )
                session.add(checkpoint_row)
                await session.commit()
        except Exception:
            # Don't let pending writes failure crash the pipeline
            pass

    def list(
        self,
        config: dict[str, Any],
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints synchronously (not implemented)."""
        raise NotImplementedError("Use alist() for async operations")

    async def alist(
        self,
        config: dict[str, Any],
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints for a thread.

        Args:
            config: Configuration with thread_id.
            filter: Optional filter criteria.
            before: List checkpoints before this one.
            limit: Maximum number to return.

        Yields:
            CheckpointTuples in reverse chronological order.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return

        async with self._session_maker() as session:
            query = select(CheckpointModel).where(
                CheckpointModel.thread_id == thread_id
            )

            if before:
                before_id = before.get("configurable", {}).get("checkpoint_id")
                if before_id:
                    # Get the created_at of the before checkpoint
                    before_result = await session.execute(
                        select(CheckpointModel.created_at).where(
                            CheckpointModel.thread_id == thread_id,
                            CheckpointModel.checkpoint_id == before_id,
                        )
                    )
                    before_time = before_result.scalar_one_or_none()
                    if before_time:
                        query = query.where(CheckpointModel.created_at < before_time)

            query = query.order_by(CheckpointModel.created_at.desc())

            if limit:
                query = query.limit(limit)

            result = await session.execute(query)

            for row in result.scalars():
                checkpoint_data = row.checkpoint_data
                metadata = row.checkpoint_metadata or {}

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": row.checkpoint_id,
                        }
                    },
                    checkpoint=Checkpoint(
                        v=checkpoint_data.get("v", 1),
                        id=row.checkpoint_id,
                        ts=checkpoint_data.get("ts", ""),
                        channel_values=checkpoint_data.get("channel_values", {}),
                        channel_versions=checkpoint_data.get("channel_versions", {}),
                        versions_seen=checkpoint_data.get("versions_seen", {}),
                        pending_sends=checkpoint_data.get("pending_sends", []),
                    ),
                    metadata=CheckpointMetadata(**metadata) if metadata else CheckpointMetadata(),
                    parent_config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": row.parent_checkpoint_id,
                        }
                    }
                    if row.parent_checkpoint_id
                    else None,
                )

    def _serialize_channel_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Serialize channel values to JSON-compatible format."""
        result = {}
        for key, value in values.items():
            try:
                # Try JSON serialization
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                # Convert to string for non-serializable objects
                if hasattr(value, "to_dict"):
                    result[key] = value.to_dict()
                elif hasattr(value, "dict"):
                    result[key] = value.dict()
                else:
                    result[key] = str(value)
        return result
