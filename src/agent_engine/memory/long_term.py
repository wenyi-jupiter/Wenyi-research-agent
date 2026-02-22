"""Long-term memory using pgvector."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Mapped, mapped_column

from agent_engine.config import get_settings
from agent_engine.memory.embeddings import BaseEmbeddingProvider, get_embedding_provider
from agent_engine.persistence.models import Base


class MemoryEntry(Base):
    """Vector memory entry model."""

    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(32), default="text", index=True
    )  # text, summary, result, etc.
    # Note: Vector dimension should match EMBEDDING_DIMENSION in .env
    # text-embedding-v4 default is 1024, but supports: 2048, 1536, 1024, 768, 512, 256, 128, 64
    embedding: Mapped[Any] = mapped_column(Vector(1024), nullable=False)
    entry_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, default=None)
    importance: Mapped[float] = mapped_column(Float, default=0.5)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index(
            "ix_memory_entries_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


@dataclass
class MemorySearchResult:
    """Result from a memory search."""

    id: int
    content: str
    content_type: str
    similarity: float
    task_id: str | None
    metadata: dict | None
    created_at: datetime


class LongTermMemory:
    """Long-term memory store using pgvector."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
        session_maker: async_sessionmaker[AsyncSession] | None = None,
    ):
        """Initialize long-term memory.

        Args:
            embedding_provider: Provider for generating embeddings.
            session_maker: Optional pre-configured session maker.
        """
        self.embedding_provider = embedding_provider or get_embedding_provider()

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

    async def store(
        self,
        content: str,
        content_type: str = "text",
        task_id: str | None = None,
        metadata: dict | None = None,
        importance: float = 0.5,
        expires_at: datetime | None = None,
    ) -> int:
        """Store content in long-term memory.

        Args:
            content: Text content to store.
            content_type: Type of content (text, summary, result).
            task_id: Associated task ID.
            metadata: Additional metadata.
            importance: Importance score (0-1).
            expires_at: Optional expiration time.

        Returns:
            ID of the created memory entry.
        """
        # Generate embedding
        embedding = await self.embedding_provider.embed_text(content)

        async with self._session_maker() as session:
            entry = MemoryEntry(
                content=content,
                content_type=content_type,
                task_id=task_id,
                embedding=embedding,
                entry_metadata=metadata,
                importance=importance,
                expires_at=expires_at,
            )
            session.add(entry)
            await session.commit()
            await session.refresh(entry)
            return entry.id

    async def store_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> list[int]:
        """Store multiple entries in long-term memory.

        Args:
            entries: List of entry dictionaries with content, content_type, etc.

        Returns:
            List of created entry IDs.
        """
        # Generate embeddings in batch
        contents = [e["content"] for e in entries]
        embeddings = await self.embedding_provider.embed_texts(contents)

        async with self._session_maker() as session:
            ids = []
            for entry_data, embedding in zip(entries, embeddings):
                entry = MemoryEntry(
                    content=entry_data["content"],
                    content_type=entry_data.get("content_type", "text"),
                    task_id=entry_data.get("task_id"),
                    embedding=embedding,
                    entry_metadata=entry_data.get("metadata"),
                    importance=entry_data.get("importance", 0.5),
                    expires_at=entry_data.get("expires_at"),
                )
                session.add(entry)
                await session.flush()
                ids.append(entry.id)

            await session.commit()
            return ids

    async def search(
        self,
        query: str,
        limit: int = 5,
        content_type: str | None = None,
        task_id: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[MemorySearchResult]:
        """Search for similar memories.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            content_type: Filter by content type.
            task_id: Filter by task ID.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of search results sorted by similarity.
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)

        async with self._session_maker() as session:
            # Use cosine similarity
            similarity = 1 - MemoryEntry.embedding.cosine_distance(query_embedding)

            query_stmt = (
                select(
                    MemoryEntry.id,
                    MemoryEntry.content,
                    MemoryEntry.content_type,
                    MemoryEntry.task_id,
                    MemoryEntry.entry_metadata,
                    MemoryEntry.created_at,
                    similarity.label("similarity"),
                )
                .where(similarity >= min_similarity)
                .order_by(similarity.desc())
                .limit(limit)
            )

            if content_type:
                query_stmt = query_stmt.where(MemoryEntry.content_type == content_type)

            if task_id:
                query_stmt = query_stmt.where(MemoryEntry.task_id == task_id)

            result = await session.execute(query_stmt)

            return [
                MemorySearchResult(
                    id=row.id,
                    content=row.content,
                    content_type=row.content_type,
                    similarity=float(row.similarity),
                    task_id=row.task_id,
                    metadata=row.entry_metadata,
                    created_at=row.created_at,
                )
                for row in result
            ]

    async def get_context(
        self,
        query: str,
        limit: int = 5,
        max_tokens: int = 2000,
    ) -> str:
        """Get relevant context for a query.

        Args:
            query: The query to find context for.
            limit: Maximum memories to retrieve.
            max_tokens: Approximate maximum tokens in context.

        Returns:
            Formatted context string.
        """
        results = await self.search(query, limit=limit, min_similarity=0.3)

        if not results:
            return ""

        context_parts = []
        total_chars = 0
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4

        for result in results:
            if total_chars + len(result.content) > max_chars:
                break

            context_parts.append(
                f"[{result.content_type}] (similarity: {result.similarity:.2f})\n{result.content}"
            )
            total_chars += len(result.content)

        return "\n\n---\n\n".join(context_parts)

    async def delete(self, entry_id: int) -> bool:
        """Delete a memory entry.

        Args:
            entry_id: The entry ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        async with self._session_maker() as session:
            result = await session.execute(
                delete(MemoryEntry).where(MemoryEntry.id == entry_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def delete_by_task(self, task_id: str) -> int:
        """Delete all memories for a task.

        Args:
            task_id: The task ID.

        Returns:
            Number of deleted entries.
        """
        async with self._session_maker() as session:
            result = await session.execute(
                delete(MemoryEntry).where(MemoryEntry.task_id == task_id)
            )
            await session.commit()
            return result.rowcount

    async def cleanup_expired(self) -> int:
        """Delete expired memory entries.

        Returns:
            Number of deleted entries.
        """
        async with self._session_maker() as session:
            result = await session.execute(
                delete(MemoryEntry).where(
                    MemoryEntry.expires_at.isnot(None),
                    MemoryEntry.expires_at < datetime.utcnow(),
                )
            )
            await session.commit()
            return result.rowcount

    async def clear_all(self) -> int:
        """Delete all memory entries.

        Returns:
            Number of deleted entries.
        """
        async with self._session_maker() as session:
            result = await session.execute(delete(MemoryEntry))
            await session.commit()
            return result.rowcount


# Global memory instance
_memory: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """Get the global long-term memory instance."""
    global _memory
    if _memory is None:
        _memory = LongTermMemory()
    return _memory
