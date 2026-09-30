"""Tests for Quality Guard."""

import pytest
from edgecascade.fallback.quality import QualityGuard


def test_quality_valid_response():
    guard = QualityGuard()
    text = "The quick brown fox jumps over the lazy dog. This is a comprehensive answer with variety."
    res = guard.validate(text)
    assert res.is_valid
    assert not res.should_cascade


def test_quality_empty_response():
    guard = QualityGuard()
    res = guard.validate("   \n  ")
    assert not res.is_valid
    assert res.should_cascade
    assert "Empty" in res.failure_reason


def test_quality_too_short():
    guard = QualityGuard(min_tokens=3)
    res = guard.validate("Hi")
    assert not res.is_valid
    assert res.should_cascade
    assert "too short" in res.failure_reason


def test_quality_degenerate_loop():
    guard = QualityGuard(max_repetition_ratio=0.35)
    # Stuttering repetitive output
    text = "repeat this word repeat this word repeat this word repeat this word repeat this word repeat this word repeat this word"
    res = guard.validate(text)
    assert not res.is_valid
    assert res.should_cascade
    assert "repetition" in res.failure_reason.lower()


def test_quality_refusal_detected():
    guard = QualityGuard()
    text = "I am sorry, as a small language model I do not have the capacity to answer this query."
    res = guard.validate(text)
    assert not res.is_valid
    assert res.should_cascade
    assert "refusal" in res.failure_reason.lower() or "inability" in res.failure_reason.lower()


def test_quality_json_expected():
    guard = QualityGuard()
    res_bad = guard.validate("Here is the answer: not a valid json", json_expected=True)
    assert not res_bad.is_valid
    assert res_bad.should_cascade

    res_good = guard.validate('```json\n{"status": "ok", "code": 200}\n```', json_expected=True)
    assert res_good.is_valid
    assert not res_good.should_cascade
