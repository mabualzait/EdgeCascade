"""Base interface for all LLM / SLM inference providers."""

from __future__ import annotations

import abc
from typing import AsyncIterator, Optional
import httpx
from edgecascade.api.schemas import ChatCompletionRequest, ChatCompletionResponse


class BaseProvider(abc.ABC):
    """Abstract base class for all EdgeCascade backend inference providers."""

    def __init__(
        self,
        name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.name = name
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Shared asynchronous HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

    def get_client(self) -> httpx.AsyncClient:
        """Get or initialize shared httpx async client."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
                limits=limits,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client session."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @abc.abstractmethod
    async def chat_complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Execute a non-streaming chat completion."""
        pass

    @abc.abstractmethod
    async def chat_complete_stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[str]:
        """Execute a streaming chat completion yielding SSE chunks."""
        pass

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend provider is reachable and healthy."""
        pass
