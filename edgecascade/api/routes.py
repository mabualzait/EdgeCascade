"""FastAPI routes for OpenAI-compatible reverse proxy, health, and metrics."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from edgecascade.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from edgecascade.config import EdgeCascadeConfig, load_config
from edgecascade.router.engine import RoutingEngine

logger = logging.getLogger("edgecascade.api")


def create_app(config: Optional[EdgeCascadeConfig] = None) -> FastAPI:
    """Create and configure EdgeCascade FastAPI application."""
    if config is None:
        config = load_config()

    engine = RoutingEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("EdgeCascade inference router initialized successfully.")
        yield
        logger.info("Shutting down EdgeCascade inference router...")
        await engine.close()

    app = FastAPI(
        title="EdgeCascade Router",
        description="Semantic Fallback Router & Hybrid Inference Engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS configuration to enable direct browser access from Cursor, Open WebUI, and Web dashboards
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication dependency
    async def verify_auth(authorization: Optional[str] = Header(None)) -> None:
        expected_key = config.server.api_key
        if not expected_key:
            return  # Authentication disabled

        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )
        token = authorization.replace("Bearer ", "").strip()
        if token != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
    async def chat_completions(req: ChatCompletionRequest):
        """OpenAI-compatible chat completions proxy endpoint with smart routing & fallback."""
        if req.stream:
            stream_gen = engine.route_stream(req)
            return StreamingResponse(
                stream_gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "x-edgecascade-version": "0.1.0",
                },
            )

        try:
            response: ChatCompletionResponse = await engine.route_completion(req)
            resp_dict = response.model_dump()

            headers = {
                "x-edgecascade-tier": str(response.routed_tier),
                "x-edgecascade-provider": str(response.routed_provider),
                "x-edgecascade-fallback": str(response.fallback_triggered).lower(),
                "x-edgecascade-privacy-contained": str(response.privacy_contained).lower(),
                "x-edgecascade-routing-time-ms": f"{response.routing_latency_ms:.2f}" if response.routing_latency_ms else "0.0",
                "x-edgecascade-cost-saved-usd": f"{response.cost_saved_usd:.6f}" if response.cost_saved_usd else "0.000000",
            }
            return JSONResponse(content=resp_dict, headers=headers)

        except Exception as e:
            logger.error("Chat completion routing failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Inference cascade failed across all providers: {str(e)}",
            )

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible list of available virtual and target models."""
        models = [
            {"id": "edgecascade-auto", "object": "model", "owned_by": "edgecascade"},
        ]
        # Include models defined in configured tiers
        for tier_id, t_cfg in config.tiers.items():
            models.append({"id": t_cfg.primary_model, "object": "model", "owned_by": t_cfg.primary_provider})
            if t_cfg.fallback_model:
                models.append({"id": t_cfg.fallback_model, "object": "model", "owned_by": t_cfg.fallback_provider or "cloud"})

        # Deduplicate
        seen = set()
        unique_models = []
        for m in models:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique_models.append(m)

        return {"object": "list", "data": unique_models}

    @app.get("/health")
    async def health():
        """Health and readiness probe."""
        provider_statuses: Dict[str, bool] = {}
        for p_name, provider in engine.providers.items():
            try:
                provider_statuses[p_name] = await provider.health_check()
            except Exception:
                provider_statuses[p_name] = False

        cb_statuses = {
            name: cb.get_status() for name, cb in engine.circuit_breakers.items()
        }

        return {
            "status": "healthy",
            "uptime_seconds": int(engine.metrics.get_summary()["uptime_seconds"]),
            "providers": provider_statuses,
            "circuit_breakers": cb_statuses,
        }

    @app.get("/metrics")
    async def metrics():
        """Expose standard Prometheus metrics endpoint."""
        body = engine.metrics.export_prometheus()
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/api/stats")
    async def stats():
        """JSON summary of telemetry, latency percentiles, and cost savings."""
        return engine.metrics.get_summary()

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve rich visual observability dashboard."""
        html_path = Path(__file__).resolve().parent.parent / "web" / "templates" / "dashboard.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>EdgeCascade Dashboard Template Not Found</h1>", status_code=404)

    @app.get("/")
    async def root():
        """Root redirect / info."""
        return {
            "service": "EdgeCascade",
            "version": "0.1.0",
            "status": "online",
            "endpoints": {
                "completions": "/v1/chat/completions",
                "models": "/v1/models",
                "dashboard": "/dashboard",
                "stats": "/api/stats",
                "metrics": "/metrics",
                "health": "/health",
            },
        }

    return app
