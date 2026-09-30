"""Ollama provider implementation for local SLMs."""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional
import httpx
from edgecascade.api.schemas import (
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from edgecascade.providers.base import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider adapter for local Ollama instances."""

    def __init__(
        self,
        name: str = "ollama",
        base_url: Optional[str] = "http://localhost:11434",
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(name, base_url, api_key, timeout_seconds, max_retries)

    async def health_check(self) -> bool:
        """Check if Ollama server is responding."""
        try:
            client = self.get_client()
            url = f"{self.base_url}/api/tags"
            resp = await client.get(url, timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def chat_complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Execute chat completion using Ollama's OpenAI-compatible endpoint."""
        client = self.get_client()
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Parse choices
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(
                ChatCompletionChoice(
                    index=c.get("index", 0),
                    message=ChatMessage(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content", ""),
                    ),
                    finish_reason=c.get("finish_reason", "stop"),
                )
            )

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=data.get("id", "ollama-resp"),
            model=model,
            choices=choices,
            usage=usage,
        )

    async def chat_complete_stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[str]:
        """Stream SSE chunks directly from Ollama."""
        client = self.get_client()
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"
