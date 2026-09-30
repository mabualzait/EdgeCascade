"""Core Routing & Fallback Engine for EdgeCascade.

Orchestrates the Privacy Gate, Semantic Scorer, Circuit Breakers,
Unified Providers, Quality Guard, and Observability Metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple

from edgecascade.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from edgecascade.circuit_breaker.breaker import CircuitBreaker, CircuitState
from edgecascade.classifier.scorer import ComplexityScorer
from edgecascade.config import EdgeCascadeConfig, TierConfig
from edgecascade.fallback.quality import QualityGuard
from edgecascade.metrics.collector import MetricsCollector
from edgecascade.pii.detector import PIIDetector, PIIDetectionResult
from edgecascade.providers.anthropic import AnthropicProvider
from edgecascade.providers.base import BaseProvider
from edgecascade.providers.gemini import GeminiProvider
from edgecascade.providers.mock import MockProvider
from edgecascade.providers.ollama import OllamaProvider
from edgecascade.providers.openai_provider import OpenAIProvider
from edgecascade.providers.vllm import VLLMProvider

logger = logging.getLogger("edgecascade.router")


class RoutingEngine:
    """Central engine routing inference requests across edge and cloud tiers."""

    def __init__(self, config: EdgeCascadeConfig) -> None:
        self.config = config

        # 1. Privacy Gate
        pg = config.privacy_gate
        self.pii_detector = PIIDetector(
            detect_pii=pg.detect_pii,
            detect_secrets=pg.detect_secrets,
            detect_code_secrets=pg.detect_code_secrets,
            enforce_local_only=pg.enforce_local_only,
        )

        # 2. Semantic Complexity Scorer
        cs = config.complexity_scorer
        self.complexity_scorer = ComplexityScorer(
            tier_1_max_threshold=cs.tier_1_max_threshold,
            tier_2_max_threshold=cs.tier_2_max_threshold,
            weights=cs.weights.model_dump(),
        )

        # 3. Quality Guard
        qg = config.quality_guard
        self.quality_guard = QualityGuard(
            min_tokens=qg.min_tokens,
            repetition_check=qg.repetition_check,
            max_repetition_ratio=qg.max_repetition_ratio,
            refusal_check=qg.refusal_check,
        )

        # 4. Metrics & Observability Collector
        self.metrics = MetricsCollector(cost_config=config.cost_metrics)

        # 5. Initialize Providers & Circuit Breakers
        self.providers: Dict[str, BaseProvider] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        """Instantiate configured providers and circuit breakers."""
        cb_cfg = self.config.circuit_breaker

        for p_name, p_cfg in self.config.providers.items():
            # Build provider
            provider: Optional[BaseProvider] = None
            ptype = p_cfg.type.lower()

            if ptype == "ollama":
                provider = OllamaProvider(
                    name=p_name,
                    base_url=p_cfg.base_url,
                    api_key=p_cfg.api_key,
                    timeout_seconds=p_cfg.timeout_seconds,
                    max_retries=p_cfg.max_retries,
                )
            elif ptype == "vllm":
                provider = VLLMProvider(
                    name=p_name,
                    base_url=p_cfg.base_url,
                    api_key=p_cfg.api_key,
                    timeout_seconds=p_cfg.timeout_seconds,
                    max_retries=p_cfg.max_retries,
                )
            elif ptype == "openai":
                provider = OpenAIProvider(
                    name=p_name,
                    base_url=p_cfg.base_url,
                    api_key=p_cfg.api_key,
                    timeout_seconds=p_cfg.timeout_seconds,
                    max_retries=p_cfg.max_retries,
                )
            elif ptype == "anthropic":
                provider = AnthropicProvider(
                    name=p_name,
                    base_url=p_cfg.base_url,
                    api_key=p_cfg.api_key,
                    timeout_seconds=p_cfg.timeout_seconds,
                    max_retries=p_cfg.max_retries,
                )
            elif ptype == "gemini":
                provider = GeminiProvider(
                    name=p_name,
                    base_url=p_cfg.base_url,
                    api_key=p_cfg.api_key,
                    timeout_seconds=p_cfg.timeout_seconds,
                    max_retries=p_cfg.max_retries,
                )
            elif ptype == "mock":
                provider = MockProvider(
                    name=p_name,
                    latency_ms=p_cfg.latency_ms or 5.0,
                )

            if provider:
                self.providers[p_name] = provider
                self.circuit_breakers[p_name] = CircuitBreaker(
                    name=p_name,
                    failure_threshold=cb_cfg.failure_threshold,
                    recovery_timeout_seconds=cb_cfg.recovery_timeout_seconds,
                    half_open_max_trials=cb_cfg.half_open_max_trials,
                )

    def _is_cloud_provider(self, provider_name: str) -> bool:
        """Check if provider is external/cloud based."""
        p_cfg = self.config.providers.get(provider_name)
        if not p_cfg:
            return False
        return p_cfg.type.lower() in ("openai", "anthropic", "gemini")

    def evaluate_routing(
        self, request: ChatCompletionRequest
    ) -> Tuple[str, PIIDetectionResult, Optional[float]]:
        """Evaluate privacy and semantic complexity to determine target tier."""
        routing_start = time.perf_counter()

        # Step 1: Privacy & PII Evaluation
        pii_result = self.pii_detector.scan_messages([m.model_dump() for m in request.messages])

        # Step 2: Override checks
        if request.force_tier and request.force_tier in self.config.tiers:
            # If privacy requires local, ensure forced tier doesn't send to cloud
            if pii_result.requires_local_routing:
                tier_cfg = self.config.tiers[request.force_tier]
                if self._is_cloud_provider(tier_cfg.primary_provider):
                    logger.warning(
                        "Caller forced tier %s, but sensitive data detected. Overriding to local tier_1",
                        request.force_tier,
                    )
                    return "tier_1", pii_result, (time.perf_counter() - routing_start) * 1000.0
            return request.force_tier, pii_result, (time.perf_counter() - routing_start) * 1000.0

        # Step 3: If privacy rule triggered, restrict strictly to local SLM
        if pii_result.requires_local_routing:
            logger.info("Sensitive data detected (%s). Routing strictly to local tier_1", pii_result.reason)
            return "tier_1", pii_result, (time.perf_counter() - routing_start) * 1000.0

        # Step 4: Semantic Complexity Scoring
        score_res = self.complexity_scorer.score_messages([m.model_dump() for m in request.messages])
        routing_latency_ms = (time.perf_counter() - routing_start) * 1000.0

        logger.debug(
            "Scored query: score=%.3f -> %s (eval time: %.2f ms)",
            score_res.score,
            score_res.tier,
            routing_latency_ms,
        )
        return score_res.tier, pii_result, routing_latency_ms

    def _get_cascade_plan(
        self, initial_tier: str, privacy_locked: bool
    ) -> List[Tuple[str, str, str]]:
        """Compute ordered list of (tier, provider_name, model_name) to attempt."""
        plan: List[Tuple[str, str, str]] = []

        # Tier progression order
        tiers_order = ["tier_1", "tier_2", "tier_3"]
        start_idx = tiers_order.index(initial_tier) if initial_tier in tiers_order else 0

        for t in tiers_order[start_idx:]:
            t_cfg: Optional[TierConfig] = self.config.tiers.get(t)
            if not t_cfg:
                continue

            # Primary provider
            if not (privacy_locked and self._is_cloud_provider(t_cfg.primary_provider)):
                plan.append((t, t_cfg.primary_provider, t_cfg.primary_model))

            # Fallback provider within tier
            if t_cfg.fallback_provider and t_cfg.fallback_model:
                if not (privacy_locked and self._is_cloud_provider(t_cfg.fallback_provider)):
                    plan.append((t, t_cfg.fallback_provider, t_cfg.fallback_model))

        # If plan is empty because of strict privacy lock, find first local provider
        if not plan:
            for t in ["tier_1", "tier_2"]:
                t_cfg = self.config.tiers.get(t)
                if t_cfg and not self._is_cloud_provider(t_cfg.primary_provider):
                    plan.append((t, t_cfg.primary_provider, t_cfg.primary_model))
                    break

        return plan

    async def route_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Route non-streaming request with automatic cascading and quality guard."""
        req_start = time.perf_counter()
        target_tier, pii_result, routing_latency_ms = self.evaluate_routing(request)
        cascade_plan = self._get_cascade_plan(target_tier, pii_result.requires_local_routing)

        last_error: Optional[Exception] = None
        fallback_occurred = False
        fallback_reason: Optional[str] = None

        for attempt_idx, (tier_name, provider_name, model_name) in enumerate(cascade_plan):
            provider = self.providers.get(provider_name)
            cb = self.circuit_breakers.get(provider_name)

            if not provider or not cb:
                logger.warning("Provider [%s] not found in active providers", provider_name)
                continue

            # Check circuit breaker
            if not await cb.can_execute():
                logger.warning(
                    "Circuit breaker for [%s] is OPEN. Skipping to next cascade tier.",
                    provider_name,
                )
                fallback_occurred = True
                fallback_reason = f"Circuit breaker OPEN for {provider_name}"
                continue

            if attempt_idx > 0:
                fallback_occurred = True

            try:
                # Execute inference
                logger.info(
                    "Executing inference on tier [%s], provider [%s], model [%s]",
                    tier_name,
                    provider_name,
                    model_name,
                )
                response = await provider.chat_complete(request, model_name)
                await cb.record_success()

                # Check Output Quality Guard (if enabled and non-empty response)
                if self.config.quality_guard.enabled and response.choices:
                    content = response.choices[0].message.content
                    text_content = content if isinstance(content, str) else str(content)
                    q_res = self.quality_guard.validate(text_content)
                    if not q_res.is_valid and q_res.should_cascade:
                        logger.warning(
                            "Quality guard failed on [%s]: %s. Cascading upward...",
                            model_name,
                            q_res.failure_reason,
                        )
                        fallback_occurred = True
                        fallback_reason = f"Quality guard: {q_res.failure_reason}"
                        # Try next option in cascade plan
                        continue

                # Successful execution
                total_latency_ms = (time.perf_counter() - req_start) * 1000.0
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0

                cost_saved = self.metrics.record_request(
                    tier=tier_name,
                    provider=provider_name,
                    model=model_name,
                    latency_ms=total_latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    privacy_contained=pii_result.requires_local_routing,
                    privacy_categories=pii_result.categories,
                    fallback_occurred=fallback_occurred,
                    fallback_reason=fallback_reason,
                    status="success",
                )

                # Attach EdgeCascade routing metadata
                response.routed_tier = tier_name
                response.routed_provider = provider_name
                response.fallback_triggered = fallback_occurred
                response.privacy_contained = pii_result.requires_local_routing
                response.routing_latency_ms = routing_latency_ms
                response.cost_saved_usd = cost_saved

                return response

            except Exception as e:
                logger.warning(
                    "Provider [%s] failed with %s: %s. Recording failure and cascading.",
                    provider_name,
                    type(e).__name__,
                    str(e),
                )
                await cb.record_failure(str(e))
                last_error = e
                fallback_occurred = True
                fallback_reason = f"{type(e).__name__}: {str(e)}"
                continue

        # All cascade options failed
        total_latency_ms = (time.perf_counter() - req_start) * 1000.0
        self.metrics.record_request(
            tier=target_tier,
            provider="none",
            model="none",
            latency_ms=total_latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            privacy_contained=pii_result.requires_local_routing,
            status="error",
        )
        raise RuntimeError(
            f"All cascade routes failed. Last error: {type(last_error).__name__}: {last_error}"
        )

    async def route_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Route streaming request through selected provider."""
        req_start = time.perf_counter()
        target_tier, pii_result, routing_latency_ms = self.evaluate_routing(request)
        cascade_plan = self._get_cascade_plan(target_tier, pii_result.requires_local_routing)

        last_error: Optional[Exception] = None

        for tier_name, provider_name, model_name in cascade_plan:
            provider = self.providers.get(provider_name)
            cb = self.circuit_breakers.get(provider_name)

            if not provider or not cb:
                continue

            if not await cb.can_execute():
                continue

            try:
                stream_iter = provider.chat_complete_stream(request, model_name)
                # Stream first chunk to verify connection before committing
                async for chunk in stream_iter:
                    yield chunk

                await cb.record_success()
                total_latency_ms = (time.perf_counter() - req_start) * 1000.0
                self.metrics.record_request(
                    tier=tier_name,
                    provider=provider_name,
                    model=model_name,
                    latency_ms=total_latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    privacy_contained=pii_result.requires_local_routing,
                    privacy_categories=pii_result.categories,
                    status="success",
                )
                return

            except Exception as e:
                logger.warning("Streaming error on [%s]: %s", provider_name, e)
                await cb.record_failure(str(e))
                last_error = e
                continue

        raise RuntimeError(f"All streaming routes failed. Last error: {last_error}")

    async def close(self) -> None:
        """Gracefully close all provider HTTP sessions."""
        for provider in self.providers.values():
            await provider.close()
