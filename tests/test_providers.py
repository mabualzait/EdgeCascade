"""Tests for Provider Adapters."""

import pytest
from edgecascade.api.schemas import ChatCompletionRequest, ChatMessage
from edgecascade.providers.mock import MockProvider
from edgecascade.providers.anthropic import AnthropicProvider
from edgecascade.providers.gemini import GeminiProvider


@pytest.mark.asyncio
async def test_mock_provider_chat_complete():
    provider = MockProvider(name="test_mock", latency_ms=1.0)
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello mock!")]
    )
    resp = await provider.chat_complete(req, model="mock-model")

    assert resp.model == "mock-model"
    assert len(resp.choices) == 1
    assert "mock" in resp.choices[0].message.content.lower()
    assert resp.usage.prompt_tokens > 0


@pytest.mark.asyncio
async def test_mock_provider_stream():
    provider = MockProvider(name="test_mock", latency_ms=1.0)
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello stream!")]
    )
    chunks = []
    async for chunk in provider.chat_complete_stream(req, model="mock-model"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert any("[DONE]" in c for c in chunks)


def test_anthropic_message_converter():
    provider = AnthropicProvider(api_key="sk-ant-test")
    req = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="You are a system prompt."),
            ChatMessage(role="user", content="User question."),
            ChatMessage(role="assistant", content="Assistant reply."),
        ]
    )
    sys_prompt, messages = provider._convert_messages(req)
    assert sys_prompt == "You are a system prompt."
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_gemini_contents_converter():
    provider = GeminiProvider(api_key="gemini-test-key")
    req = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="System instruction."),
            ChatMessage(role="user", content="Hello Gemini."),
        ]
    )
    sys_inst, contents = provider._convert_contents(req)
    assert sys_inst is not None
    assert sys_inst["parts"][0]["text"] == "System instruction."
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
