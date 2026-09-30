"""Tests for Routing Engine and Cascade Logic."""

import pytest
from edgecascade.api.schemas import ChatCompletionRequest, ChatMessage
from edgecascade.router.engine import RoutingEngine


@pytest.mark.asyncio
async def test_router_tier_1_routing(mock_config):
    engine = RoutingEngine(mock_config)
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Summarize this simple sentence in json format.")]
    )
    resp = await engine.route_completion(req)

    assert resp.routed_tier == "tier_1"
    assert resp.routed_provider == "mock_local_slm"
    assert not resp.fallback_triggered
    assert not resp.privacy_contained
    assert resp.routing_latency_ms is not None
    assert resp.cost_saved_usd is not None


@pytest.mark.asyncio
async def test_router_privacy_lock(mock_config):
    engine = RoutingEngine(mock_config)
    # High complexity query that would normally go to tier_3, BUT contains an AWS key!
    req = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="Design a Byzantine fault-tolerant consensus protocol. My AWS key is AKIA1234567890ABCDEF",
            )
        ]
    )
    resp = await engine.route_completion(req)

    # MUST be contained locally!
    assert resp.routed_tier == "tier_1"
    assert resp.routed_provider in ("mock_local_slm", "mock_local_mid")
    assert resp.privacy_contained is True


@pytest.mark.asyncio
async def test_router_fallback_on_primary_failure(mock_config):
    engine = RoutingEngine(mock_config)
    # Configure mock_local_slm to simulate failure
    mock_provider = engine.providers["mock_local_slm"]
    mock_provider.simulate_failure = True

    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Summarize this text.")]
    )
    resp = await engine.route_completion(req)

    # Primary failed, should cascade to fallback provider mock_local_mid
    assert resp.fallback_triggered is True
    assert resp.routed_provider == "mock_local_mid"


@pytest.mark.asyncio
async def test_router_fallback_on_quality_failure(mock_config):
    engine = RoutingEngine(mock_config)
    # Make primary return degenerate loop
    engine.providers["mock_local_slm"].mock_response_text = (
        "repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat"
    )

    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Classify sentiment.")]
    )
    resp = await engine.route_completion(req)

    # Quality check should have caught repetitive spam and cascaded upward
    assert resp.fallback_triggered is True
    assert resp.routed_provider == "mock_local_mid"
