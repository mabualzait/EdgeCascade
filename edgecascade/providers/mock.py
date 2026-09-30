"""Mock provider for unit testing, offline testing, and CI/CD pipelines."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterator, Optional
from edgecascade.api.schemas import (
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from edgecascade.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """In-memory mock provider for rapid local testing and simulations."""

    def __init__(
        self,
        name: str = "mock",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 1,
        latency_ms: float = 5.0,
        simulate_failure: bool = False,
        mock_response_text: Optional[str] = None,
    ) -> None:
        super().__init__(name, base_url, api_key, timeout_seconds, max_retries)
        self.latency_ms = latency_ms
        self.simulate_failure = simulate_failure
        self.mock_response_text = mock_response_text
        self.call_count = 0

    async def health_check(self) -> bool:
        return not self.simulate_failure

    async def chat_complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        self.call_count += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        if self.simulate_failure:
            raise ConnectionError(f"MockProvider [{self.name}] simulated failure")

        response_text = (
            self.mock_response_text
            or f"EdgeCascade Mock inference response from [{model}] on provider [{self.name}]."
        )

        prompt_len = sum(
            len(str(m.content).split()) for m in request.messages
        )
        completion_len = len(response_text.split())

        return ChatCompletionResponse(
            id=f"mock-{uuid.uuid4().hex[:8]}",
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=prompt_len + completion_len,
            ),
        )

    async def chat_complete_stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncIterator[str]:
        self.call_count += 1
        if self.simulate_failure:
            raise ConnectionError(f"MockProvider [{self.name}] simulated stream failure")

        words = (
            self.mock_response_text
            or f"EdgeCascade streaming mock response from [{model}]."
        ).split()

        for word in words:
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0 / max(1, len(words)))
            chunk = {
                "id": "mock-chunk",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"{word} "},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        done_chunk = {
            "id": "mock-chunk",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"
