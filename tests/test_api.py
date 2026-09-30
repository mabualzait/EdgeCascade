"""Tests for FastAPI HTTP endpoints."""

import pytest
import httpx
from edgecascade.api.routes import create_app


@pytest.mark.asyncio
async def test_health_endpoint(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "providers" in data
        assert "circuit_breakers" in data


@pytest.mark.asyncio
async def test_models_endpoint(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "edgecascade-auto" in model_ids
        assert "llama3.2:1b" in model_ids


@pytest.mark.asyncio
async def test_chat_completions_endpoint(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        req_payload = {
            "model": "edgecascade-auto",
            "messages": [{"role": "user", "content": "Extract json from this string"}],
            "temperature": 0.5,
        }
        resp = await client.post("/v1/chat/completions", json=req_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) > 0
        assert resp.headers.get("x-edgecascade-tier") == "tier_1"
        assert resp.headers.get("x-edgecascade-fallback") == "false"


@pytest.mark.asyncio
async def test_chat_completions_streaming(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        req_payload = {
            "model": "edgecascade-auto",
            "messages": [{"role": "user", "content": "Quick greeting"}],
            "stream": True,
        }
        resp = await client.post("/v1/chat/completions", json=req_payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        lines = resp.text.split("\n")
        assert any("data: " in line for line in lines)


@pytest.mark.asyncio
async def test_stats_and_metrics_endpoints(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # First trigger an inference
        await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test prompt"}]},
        )

        # Check /api/stats
        stats_resp = await client.get("/api/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_requests"] >= 1
        assert "cost_saved_usd" in stats

        # Check /metrics (Prometheus)
        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "edgecascade_requests_total" in metrics_resp.text


@pytest.mark.asyncio
async def test_dashboard_endpoint(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dashboard")
        assert resp.status_code == 200
        assert "EdgeCascade" in resp.text
