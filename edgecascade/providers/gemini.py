"""Google Gemini provider implementation adapting Gemini API to OpenAI format."""

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


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini API."""

    def __init__(
        self,
        name: str = "gemini",
        base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta",
        api_key: Optional[str] = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(name, base_url, api_key, timeout_seconds, max_retries)

    async def health_check(self) -> bool:
        """Health check returns True if API key is present."""
        return bool(self.api_key)

    def _convert_contents(self, request: ChatCompletionRequest) -> tuple[Optional[dict], List[dict]]:
        """Convert OpenAI messages to Gemini contents and system instruction."""
        system_instruction: Optional[dict] = None
        contents: List[dict] = []

        for msg in request.messages:
            text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            if msg.role == "system":
                system_instruction = {"parts": [{"text": text}]}
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": text}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        return system_instruction, contents

    async def chat_complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Call Gemini generateContent API and adapt to OpenAI format."""
        client = self.get_client()
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

        system_instruction, contents = self._convert_contents(request)

        generation_config = {
            "temperature": request.temperature,
            "topP": request.top_p,
        }
        if request.max_tokens:
            generation_config["maxOutputTokens"] = request.max_tokens

        payload: dict = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        text_output = ""
        finish_reason = "stop"
        if candidates:
            cand = candidates[0]
            parts = cand.get("content", {}).get("parts", [])
            text_output = "".join(p.get("text", "") for p in parts)
            finish_reason = cand.get("finishReason", "stop").lower()

        usage_metadata = data.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)

        choices = [
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text_output),
                finish_reason=finish_reason,
            )
        ]

        return ChatCompletionResponse(
            id=data.get("responseId", "gemini-resp"),
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
        """Stream SSE from Gemini and adapt to OpenAI SSE chunks."""
        client = self.get_client()
        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"

        system_instruction, contents = self._convert_contents(request)
        generation_config = {
            "temperature": request.temperature,
            "topP": request.top_p,
        }
        if request.max_tokens:
            generation_config["maxOutputTokens"] = request.max_tokens

        payload: dict = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                try:
                    event = json.loads(data_str)
                    candidates = event.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        delta_text = "".join(p.get("text", "") for p in parts)
                        chunk = {
                            "id": "gemini-chunk",
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
                except Exception:
                    continue

            yield "data: [DONE]\n\n"
