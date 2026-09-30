"""Pytest fixtures for EdgeCascade test suite."""

import pytest
from edgecascade.config import (
    CircuitBreakerConfig,
    ComplexityScorerConfig,
    CostMetricsConfig,
    EdgeCascadeConfig,
    PrivacyGateConfig,
    ProviderConfig,
    QualityGuardConfig,
    ServerConfig,
    TierConfig,
    TokenRate,
)


@pytest.fixture
def mock_config():
    """Returns an EdgeCascade configuration populated with fast mock providers."""
    return EdgeCascadeConfig(
        server=ServerConfig(host="127.0.0.1", port=8080),
        privacy_gate=PrivacyGateConfig(
            enabled=True,
            enforce_local_only=True,
            detect_pii=True,
            detect_secrets=True,
            detect_code_secrets=True,
        ),
        complexity_scorer=ComplexityScorerConfig(
            tier_1_max_threshold=0.35,
            tier_2_max_threshold=0.70,
        ),
        tiers={
            "tier_1": TierConfig(
                name="Edge Local SLM",
                primary_provider="mock_local_slm",
                primary_model="llama3.2:1b",
                fallback_provider="mock_local_mid",
                fallback_model="llama3.1:8b",
            ),
            "tier_2": TierConfig(
                name="Mid-tier Reasoner",
                primary_provider="mock_local_mid",
                primary_model="llama3.1:8b",
                fallback_provider="mock_cloud",
                fallback_model="gpt-4o-mini",
            ),
            "tier_3": TierConfig(
                name="Frontier Cloud",
                primary_provider="mock_cloud",
                primary_model="gpt-4o",
                fallback_provider="mock_cloud_backup",
                fallback_model="claude-3-5-sonnet",
            ),
        },
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.1,  # Fast for testing
            half_open_max_trials=1,
        ),
        quality_guard=QualityGuardConfig(
            enabled=True,
            min_tokens=2,
            repetition_check=True,
            max_repetition_ratio=0.40,
            refusal_check=True,
        ),
        providers={
            "mock_local_slm": ProviderConfig(type="mock", latency_ms=1.0),
            "mock_local_mid": ProviderConfig(type="mock", latency_ms=1.0),
            "mock_cloud": ProviderConfig(type="mock", latency_ms=1.0),
            "mock_cloud_backup": ProviderConfig(type="mock", latency_ms=1.0),
        },
        cost_metrics=CostMetricsConfig(
            benchmark_frontier_model="gpt-4o",
            rates_per_million_tokens={
                "gpt-4o": TokenRate(input=2.50, output=10.00),
                "gpt-4o-mini": TokenRate(input=0.15, output=0.60),
                "llama3.2:1b": TokenRate(input=0.0, output=0.0),
                "llama3.1:8b": TokenRate(input=0.0, output=0.0),
            },
        ),
    )
