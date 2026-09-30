"""Configuration models and loader for EdgeCascade."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "info"
    api_key: Optional[str] = None


class PrivacyGateConfig(BaseModel):
    enabled: bool = True
    enforce_local_only: bool = True
    detect_pii: bool = True
    detect_secrets: bool = True
    detect_code_secrets: bool = True
    mask_in_logs: bool = True


class ComplexityWeights(BaseModel):
    length: float = 0.15
    reasoning_markers: float = 0.35
    code_complexity: float = 0.20
    semantic_similarity: float = 0.30


class ComplexityScorerConfig(BaseModel):
    tier_1_max_threshold: float = 0.35
    tier_2_max_threshold: float = 0.70
    weights: ComplexityWeights = Field(default_factory=ComplexityWeights)


class TierConfig(BaseModel):
    name: str
    description: str = ""
    primary_provider: str
    primary_model: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    max_latency_ms: float = 10000.0


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0
    half_open_max_trials: int = 2
    latency_threshold_ms: float = 10000.0


class QualityGuardConfig(BaseModel):
    enabled: bool = True
    min_tokens: int = 2
    repetition_check: bool = True
    max_repetition_ratio: float = 0.40
    refusal_check: bool = True


class ProviderConfig(BaseModel):
    type: str  # ollama, vllm, openai, anthropic, gemini, mock
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    latency_ms: Optional[float] = None  # for mock provider


class TokenRate(BaseModel):
    input: float = 0.0
    output: float = 0.0


class CostMetricsConfig(BaseModel):
    benchmark_frontier_model: str = "gpt-4o"
    rates_per_million_tokens: Dict[str, TokenRate] = Field(default_factory=dict)


class EdgeCascadeConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    privacy_gate: PrivacyGateConfig = Field(default_factory=PrivacyGateConfig)
    complexity_scorer: ComplexityScorerConfig = Field(default_factory=ComplexityScorerConfig)
    tiers: Dict[str, TierConfig] = Field(default_factory=dict)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    quality_guard: QualityGuardConfig = Field(default_factory=QualityGuardConfig)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    cost_metrics: CostMetricsConfig = Field(default_factory=CostMetricsConfig)


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively replace ${VAR} patterns with environment variables."""
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replace_fn(match: re.Match) -> str:
            var_name = match.group(1)
            default = ""
            if ":" in var_name:
                var_name, default = var_name.split(":", 1)
            return os.environ.get(var_name, default)

        return pattern.sub(replace_fn, obj)
    elif isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    return obj


def load_config(config_path: Optional[str] = None) -> EdgeCascadeConfig:
    """Load and validate EdgeCascade configuration from YAML file or environment."""
    if config_path is None:
        config_path = os.getenv("EDGECASCADE_CONFIG", "config/config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        # Search relative to current working directory or package root
        candidates = [
            Path.cwd() / path,
            Path(__file__).resolve().parent.parent / path,
            Path("/etc/edgecascade/config.yaml"),
        ]
        resolved = None
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
        path = resolved if resolved else candidates[0]

    if not path.exists():
        # Fallback to default in-memory config if no file found
        return EdgeCascadeConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    processed_data = _substitute_env_vars(raw_data)
    return EdgeCascadeConfig(**processed_data)
