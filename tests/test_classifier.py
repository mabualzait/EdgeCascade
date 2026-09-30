"""Tests for Semantic Complexity Scorer."""

import pytest
from edgecascade.classifier.scorer import ComplexityScorer


def test_scorer_tier_1_simple_extraction():
    scorer = ComplexityScorer(tier_1_max_threshold=0.35, tier_2_max_threshold=0.70)
    prompt = "Summarize this paragraph and extract the sentiment as json."
    res = scorer.score(prompt)

    assert res.tier == "tier_1"
    assert res.score < 0.35
    assert res.latency_ms < 20.0  # Must be sub-20ms
    assert "length" in res.factors
    assert "reasoning_markers" in res.factors


def test_scorer_tier_2_coding_query():
    scorer = ComplexityScorer(tier_1_max_threshold=0.35, tier_2_max_threshold=0.70)
    prompt = """
    Write a FastAPI route that accepts a Pydantic model, validates user input,
    and returns a JSON response with status 200:
    ```python
    def process_user(u: UserModel):
        pass
    ```
    """
    res = scorer.score(prompt)
    assert res.tier in ("tier_2", "tier_3")
    assert res.score >= 0.35
    assert res.latency_ms < 20.0


def test_scorer_tier_3_high_complexity():
    scorer = ComplexityScorer(tier_1_max_threshold=0.35, tier_2_max_threshold=0.70)
    prompt = (
        "Design a distributed consensus protocol and formally verify its deadlock-free guarantees "
        "under Byzantine fault tolerance. Derive the mathematical proof of soundness and completeness "
        "and analyze asymptotic bounds under network partitions."
    )
    res = scorer.score(prompt)
    assert res.tier == "tier_3"
    assert res.score >= 0.70
    assert res.latency_ms < 20.0


def test_scorer_score_messages():
    scorer = ComplexityScorer()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Translate this to Spanish: Good morning!"},
    ]
    res = scorer.score_messages(messages)
    assert res.tier == "tier_1"
