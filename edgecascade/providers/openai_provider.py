"""OpenAI provider implementation."""

from __future__ import annotations

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


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI Cloud API."""

    def __init__(
        self,
        name: str = "openai",
        base_url: Optional[str] = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(name, base_url, api_key, timeout_seconds, max_retries)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def health_check(self) -> bool:
        """Check OpenAI connection."""
        if not self.api_key:
            return False
        try:
            client = self.get_client()
            url = f"{self.base_url}/models"
            resp = await client.get(url, headers=self._headers(), timeout=4.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def chat_complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Call OpenAI chat completions API."""
        client = self.get_client()
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        resp = await client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()

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
            id=data.get("id", "openai-resp"),
            model=model,
            choices=choices,
            usage=usage,
        )

    async def chat_complete_stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[str]:
        """Stream SSE directly from OpenAI."""
        client = self.get_client()
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        async with client.stream("POST", url, json=payload, headers=self._headers()) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"
