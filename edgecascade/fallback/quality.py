"""Automated output quality guard and confidence heuristics.

Detects degenerate loops, refusal patterns, and empty outputs to trigger
automatic upward cascading before returning a bad response to the client.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QualityValidationResult:
    """Outcome of automated output quality inspection."""
    is_valid: bool
    failure_reason: Optional[str] = None
    repetition_ratio: float = 0.0
    token_count: int = 0
    should_cascade: bool = False


class QualityGuard:
    """Heuristic quality validator for model outputs."""

    def __init__(
        self,
        min_tokens: int = 2,
        repetition_check: bool = True,
        max_repetition_ratio: float = 0.40,
        refusal_check: bool = True,
    ) -> None:
        self.min_tokens = min_tokens
        self.repetition_check = repetition_check
        self.max_repetition_ratio = max_repetition_ratio
        self.refusal_check = refusal_check

        # Common SLM confusion or failure signatures
        self._refusal_patterns = re.compile(
            r"(?i)\b(?:as\s+a\s+small\s+language\s+model|"
            r"i\s+do\s+not\s+have\s+the\s+capacity\s+to\s+answer|"
            r"i\s+cannot\s+perform\s+complex\s+reasoning|"
            r"\[generation\s+failed\]|"
            r"system\s+error:\s+context\s+window\s+exceeded)\b"
        )

    def _calculate_repetition_ratio(self, tokens: List[str], n: int = 3) -> float:
        """Calculate n-gram repetition ratio (fraction of duplicate n-grams)."""
        if len(tokens) < n * 2:
            return 0.0

        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        unique_ngrams = set(ngrams)
        duplicate_count = len(ngrams) - len(unique_ngrams)
        return duplicate_count / len(ngrams)

    def validate(
        self,
        content: str,
        json_expected: bool = False,
    ) -> QualityValidationResult:
        """Inspect generated text and decide if it meets quality standards."""
        stripped = content.strip()

        # Check 1: Empty or whitespace-only response
        if not stripped:
            return QualityValidationResult(
                is_valid=False,
                failure_reason="Empty or whitespace response",
                token_count=0,
                should_cascade=True,
            )

        words = stripped.split()
        token_count = len(words)

        # Check 2: Minimum token threshold
        if token_count < self.min_tokens:
            return QualityValidationResult(
                is_valid=False,
                failure_reason=f"Output too short ({token_count} tokens < {self.min_tokens})",
                token_count=token_count,
                should_cascade=True,
            )

        # Check 3: Repetition Loop Check (detect model degenerate stuttering)
        repetition_ratio = 0.0
        if self.repetition_check and token_count >= 15:
            repetition_ratio = self._calculate_repetition_ratio(words, n=3)
            if repetition_ratio > self.max_repetition_ratio:
                return QualityValidationResult(
                    is_valid=False,
                    failure_reason=f"Degenerate repetition loop detected (ratio={repetition_ratio:.2f})",
                    repetition_ratio=repetition_ratio,
                    token_count=token_count,
                    should_cascade=True,
                )

        # Check 4: Refusal / Failure signatures
        if self.refusal_check:
            if self._refusal_patterns.search(stripped):
                return QualityValidationResult(
                    is_valid=False,
                    failure_reason="SLM self-reported inability or internal failure pattern detected",
                    repetition_ratio=repetition_ratio,
                    token_count=token_count,
                    should_cascade=True,
                )

        # Check 5: JSON validity if caller explicitly specified JSON response format
        if json_expected:
            try:
                # Attempt to extract json if enclosed in markdown
                match = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", stripped)
                json_candidate = match.group(1) if match else stripped
                json.loads(json_candidate)
            except Exception:
                return QualityValidationResult(
                    is_valid=False,
                    failure_reason="Requested JSON format could not be parsed",
                    repetition_ratio=repetition_ratio,
                    token_count=token_count,
                    should_cascade=True,
                )

        return QualityValidationResult(
            is_valid=True,
            failure_reason=None,
            repetition_ratio=repetition_ratio,
            token_count=token_count,
            should_cascade=False,
        )
