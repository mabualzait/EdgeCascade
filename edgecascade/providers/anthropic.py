"""Anthropic provider implementation adapting Messages API to OpenAI format."""

from __future__ import annotations

import json
from typing import AsyncIterator, List, Optional
import httpx
from edgecascade.api.schemas import (
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from edgecascade.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider adapter for Anthropic Claude API."""

    def __init__(
        self,
        name: str = "anthropic",
        base_url: Optional[str] = "https://api.anthropic.com/v1",
        api_key: Optional[str] = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(name, base_url, api_key, timeout_seconds, max_retries)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    async def health_check(self) -> bool:
        """Health check returns True if API key is provided."""
        return bool(self.api_key)

    def _convert_messages(self, request: ChatCompletionRequest) -> tuple[Optional[str], List[dict]]:
        """Convert OpenAI-style messages into Anthropic system prompt and message array."""
        system_prompt: Optional[str] = None
        anthropic_messages: List[dict] = []

        for msg in request.messages:
            if msg.role == "system":
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                if system_prompt:
                    system_prompt += "\n" + content
                else:
                    system_prompt = content
            else:
                anthropic_messages.append({
                    "role": "assistant" if msg.role == "assistant" else "user",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })

        # Anthropic requires at least one user message
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": "Hello"})

        return system_prompt, anthropic_messages

    async def chat_complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Call Anthropic /v1/messages API and map response to OpenAI format."""
        client = self.get_client()
        url = f"{self.base_url}/messages"

        system_prompt, messages = self._convert_messages(request)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = await client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()

        # Extract text from content blocks
        full_text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                full_text += block.get("text", "")

        anthropic_usage = data.get("usage", {})
        prompt_tokens = anthropic_usage.get("input_tokens", 0)
        completion_tokens = anthropic_usage.get("output_tokens", 0)

        choices = [
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason="stop" if data.get("stop_reason") == "end_turn" else data.get("stop_reason", "stop"),
            )
        ]

        return ChatCompletionResponse(
            id=data.get("id", "anthropic-resp"),
            model=model,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def chat_complete_stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[str]:
        """Stream Anthropic SSE and adapt to OpenAI SSE chunks."""
        client = self.get_client()
        url = f"{self.base_url}/messages"

        system_prompt, messages = self._convert_messages(request)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with client.stream("POST", url, json=payload, headers=self._headers()) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    event = json.loads(data_str)
                    event_type = event.get("type")
                    if event_type == "content_block_delta":
                        delta_text = event.get("delta", {}).get("text", "")
                        chunk = {
                            "id": "anthropic-chunk",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif event_type == "message_stop":
                        end_chunk = {
                            "id": "anthropic-chunk",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(end_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                except Exception:
                    continue
