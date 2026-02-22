"""Embedding generation for vector memory."""

from abc import ABC, abstractmethod
from typing import Any

import httpx


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        ...

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...


class MockEmbedding(BaseEmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 1536):
        """Initialize mock embedding provider.

        Args:
            dimension: Embedding dimension.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate a deterministic mock embedding."""
        import hashlib

        # Generate deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        values = []
        for i in range(0, min(len(hash_bytes), self._dimension), 1):
            values.append((hash_bytes[i % len(hash_bytes)] - 128) / 128.0)

        # Pad or truncate to dimension
        while len(values) < self._dimension:
            values.extend(values[: self._dimension - len(values)])
        return values[: self._dimension]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for multiple texts."""
        return [await self.embed_text(text) for text in texts]


class DashScopeEmbedding(BaseEmbeddingProvider):
    """Aliyun DashScope embedding provider using text-embedding-v4."""

    # Supported dimensions for text-embedding-v4
    SUPPORTED_DIMENSIONS = [2048, 1536, 1024, 768, 512, 256, 128, 64]

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimension: int = 1024,
    ):
        """Initialize DashScope embedding provider.

        Args:
            api_key: DashScope API key.
            base_url: API base URL.
            model: Model name (default: text-embedding-v4).
            dimension: Embedding dimension (default: 1024).
                       Supported: 2048, 1536, 1024, 768, 512, 256, 128, 64
        """
        if dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"Unsupported dimension: {dimension}. "
                f"Supported dimensions: {self.SUPPORTED_DIMENSIONS}"
            )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using DashScope API."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using DashScope API.
        
        Note: DashScope supports max 10 texts per request.
        """
        all_embeddings = []
        
        # Process in batches of 10 (DashScope limit)
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (max 10)."""
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self._dimension,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Extract embeddings from response
        embeddings = []
        for item in data.get("data", []):
            embeddings.append(item.get("embedding", []))

        return embeddings


def get_embedding_provider(
    provider: str | None = None,
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """Get an embedding provider instance.

    Args:
        provider: Provider name ("mock", "dashscope"). Defaults to settings.
        **kwargs: Provider-specific arguments.

    Returns:
        Embedding provider instance.
    """
    from agent_engine.config import get_settings

    settings = get_settings()
    provider = provider or settings.embedding_provider

    if provider == "mock":
        dimension = kwargs.pop("dimension", settings.embedding_dimension)
        return MockEmbedding(dimension=dimension)
    elif provider == "dashscope":
        api_key = kwargs.pop("api_key", None) or settings.dashscope_api_key
        base_url = kwargs.pop("base_url", None) or settings.dashscope_base_url
        model = kwargs.pop("model", None) or settings.embedding_model
        dimension = kwargs.pop("dimension", settings.embedding_dimension)
        
        if not api_key:
            raise ValueError("DashScope API key is required for dashscope embedding provider")
        
        return DashScopeEmbedding(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimension=dimension,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Available: mock, dashscope")
