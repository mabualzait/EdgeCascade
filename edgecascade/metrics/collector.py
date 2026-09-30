"""Observability and Cost Analytics Engine.

Calculates real-time token savings against frontier benchmark models,
tracks percentiles (p50, p95, p99), privacy firewall isolation events,
and exports standard Prometheus metrics.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional
import numpy as np
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from edgecascade.config import CostMetricsConfig


class MetricsCollector:
    """Central metrics, cost tracker, and Prometheus exporter."""

    def __init__(
        self,
        cost_config: Optional[CostMetricsConfig] = None,
        registry: Optional[CollectorRegistry] = None,
    ) -> None:
        self.cost_config = cost_config or CostMetricsConfig()
        self.registry = registry or REGISTRY

        # Rolling latency records per tier (stores last 500 samples each)
        self._tier_latencies: Dict[str, collections.deque] = {
            "tier_1": collections.deque(maxlen=500),
            "tier_2": collections.deque(maxlen=500),
            "tier_3": collections.deque(maxlen=500),
        }

        # In-memory counters
        self.total_requests = 0
        self.tier_requests: Dict[str, int] = {"tier_1": 0, "tier_2": 0, "tier_3": 0}
        self.privacy_contained_count = 0
        self.fallback_count = 0
        self.total_cost_saved_usd = 0.0
        self.total_tokens_routed = {"prompt": 0, "completion": 0}
        self.start_time = time.time()

        # Initialize Prometheus metrics (safely registering)
        self._init_prometheus()

    def _init_prometheus(self) -> None:
        """Register Prometheus metrics without duplicating."""
        try:
            self.prom_requests = Counter(
                "edgecascade_requests_total",
                "Total inference requests handled",
                ["tier", "provider", "model", "status"],
                registry=self.registry,
            )
            self.prom_latency = Histogram(
                "edgecascade_latency_seconds",
                "End-to-end request latency in seconds",
                ["tier", "provider"],
                buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
                registry=self.registry,
            )
            self.prom_fallbacks = Counter(
                "edgecascade_fallback_total",
                "Total fallback and cascade events triggered",
                ["from_tier", "reason"],
                registry=self.registry,
            )
            self.prom_privacy = Counter(
                "edgecascade_privacy_contained_total",
                "Sensitive queries blocked from cloud and isolated to local SLMs",
                ["category"],
                registry=self.registry,
            )
            self.prom_tokens = Counter(
                "edgecascade_tokens_total",
                "Total tokens processed",
                ["type", "tier"],
                registry=self.registry,
            )
            self.prom_cost_savings = Counter(
                "edgecascade_cost_savings_usd_total",
                "Cumulative USD saved compared to frontier cloud models",
                registry=self.registry,
            )
        except Exception:
            # Already registered in this process
            pass

    def calculate_cost_savings(
        self,
        actual_model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Compute USD saved by routing to actual_model instead of benchmark frontier."""
        benchmark_model = self.cost_config.benchmark_frontier_model
        rates = self.cost_config.rates_per_million_tokens

        # Benchmark cost
        bm_rate = rates.get(benchmark_model)
        if not bm_rate:
            # Fallback benchmark rates: $2.50 input / $10.00 output
            bm_input_cost = 2.50
            bm_output_cost = 10.00
        else:
            bm_input_cost = bm_rate.input
            bm_output_cost = bm_rate.output

        benchmark_cost = (
            (prompt_tokens / 1_000_000.0) * bm_input_cost
            + (completion_tokens / 1_000_000.0) * bm_output_cost
        )

        # Actual cost
        actual_rate = rates.get(actual_model)
        if not actual_rate:
            # If not in table, check if local or cheap model
            actual_input_cost = 0.0
            actual_output_cost = 0.0
        else:
            actual_input_cost = actual_rate.input
            actual_output_cost = actual_rate.output

        actual_cost = (
            (prompt_tokens / 1_000_000.0) * actual_input_cost
            + (completion_tokens / 1_000_000.0) * actual_output_cost
        )

        saved = max(0.0, benchmark_cost - actual_cost)
        return round(saved, 6)

    def record_request(
        self,
        tier: str,
        provider: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        privacy_contained: bool = False,
        privacy_categories: Optional[List[str]] = None,
        fallback_occurred: bool = False,
        fallback_reason: Optional[str] = None,
        status: str = "success",
    ) -> float:
        """Record completed request and update analytics."""
        self.total_requests += 1
        if tier in self.tier_requests:
            self.tier_requests[tier] += 1

        # Track latency
        if tier in self._tier_latencies:
            self._tier_latencies[tier].append(latency_ms)

        # Track tokens
        self.total_tokens_routed["prompt"] += prompt_tokens
        self.total_tokens_routed["completion"] += completion_tokens

        # Cost savings
        cost_saved = self.calculate_cost_savings(model, prompt_tokens, completion_tokens)
        self.total_cost_saved_usd += cost_saved

        # Privacy
        if privacy_contained:
            self.privacy_contained_count += 1
            if privacy_categories:
                for cat in privacy_categories:
                    if hasattr(self, "prom_privacy"):
                        try:
                            self.prom_privacy.labels(category=cat).inc()
                        except Exception:
                            pass

        # Fallback
        if fallback_occurred:
            self.fallback_count += 1
            if hasattr(self, "prom_fallbacks"):
                try:
                    self.prom_fallbacks.labels(
                        from_tier=tier, reason=fallback_reason or "unknown"
                    ).inc()
                except Exception:
                    pass

        # Prometheus updates
        if hasattr(self, "prom_requests"):
            try:
                self.prom_requests.labels(
                    tier=tier, provider=provider, model=model, status=status
                ).inc()
                self.prom_latency.labels(tier=tier, provider=provider).observe(
                    latency_ms / 1000.0
                )
                self.prom_tokens.labels(type="prompt", tier=tier).inc(prompt_tokens)
                self.prom_tokens.labels(type="completion", tier=tier).inc(completion_tokens)
                self.prom_cost_savings.inc(cost_saved)
            except Exception:
                pass

        return cost_saved

    def _get_percentiles(self, tier: str) -> Dict[str, float]:
        """Compute p50, p95, p99 latencies for given tier."""
        samples = list(self._tier_latencies.get(tier, []))
        if not samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(samples)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return rich summary for JSON API and dashboard."""
        uptime_seconds = int(time.time() - self.start_time)

        percentiles = {
            tier: self._get_percentiles(tier)
            for tier in ["tier_1", "tier_2", "tier_3"]
        }

        # Percentage tier distribution
        total = max(1, self.total_requests)
        distribution = {
            k: round((v / total) * 100.0, 1) for k, v in self.tier_requests.items()
        }

        return {
            "uptime_seconds": uptime_seconds,
            "total_requests": self.total_requests,
            "tier_distribution": distribution,
            "tier_counts": self.tier_requests,
            "total_tokens": self.total_tokens_routed,
            "cost_saved_usd": round(self.total_cost_saved_usd, 4),
            "privacy_contained_count": self.privacy_contained_count,
            "fallback_trigger_count": self.fallback_count,
            "latency_percentiles_ms": percentiles,
        }

    def export_prometheus(self) -> bytes:
        """Export metrics formatted for Prometheus scraping."""
        return generate_latest(self.registry)
