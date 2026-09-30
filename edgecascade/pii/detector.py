"""Deterministic Privacy, Secret & PII Gate.

Ensures that any request containing credentials, proprietary corporate code,
or personally identifiable information (PII) is deterministically locked to
local hardware (Ollama / vLLM) and never leaked to public cloud endpoints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Set


@dataclass
class PIIDetectionResult:
    """Outcome of privacy gate evaluation."""
    has_sensitive_data: bool = False
    categories: List[str] = field(default_factory=list)
    match_count: int = 0
    requires_local_routing: bool = False
    reason: str = ""


class PIIDetector:
    """High-speed deterministic regex & heuristic engine for PII and secrets."""

    def __init__(
        self,
        detect_pii: bool = True,
        detect_secrets: bool = True,
        detect_code_secrets: bool = True,
        enforce_local_only: bool = True,
    ) -> None:
        self.detect_pii = detect_pii
        self.detect_secrets = detect_secrets
        self.detect_code_secrets = detect_code_secrets
        self.enforce_local_only = enforce_local_only

        self._patterns: Dict[str, Pattern[str]] = {}
        self._build_patterns()

    def _build_patterns(self) -> None:
        """Compile regex patterns ahead of time for sub-millisecond checks."""
        patterns: Dict[str, str] = {}

        if self.detect_secrets:
            # AWS Access Key ID
            patterns["aws_access_key"] = r"\b(AKIA[0-9A-Z]{16})\b"
            # AWS Secret Key marker
            patterns["aws_secret_key"] = r"(?i)aws_secret_access_key\s*[:=]\s*['\"][A-Za-z0-9/+=]{40}['\"]"
            # Private RSA / DSA / EC / OpenSSH Keys
            patterns["private_key"] = r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PGP|ENCRYPTED)?\s*PRIVATE\s+KEY-----"
            # GitHub Personal Access Tokens
            patterns["github_token"] = r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}\b"
            # OpenAI API Keys
            patterns["openai_key"] = r"\bsk-[a-zA-Z0-9]{20,T3BlbkFJ[a-zA-Z0-9]{20,}\b|\bsk-proj-[a-zA-Z0-9_\-]{40,}\b"
            # Slack Tokens
            patterns["slack_token"] = r"\bxox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9]*\b"
            # Generic Bearer / API Key tokens
            patterns["generic_api_key"] = r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{20,}['\"]"
            # JSON Web Token (JWT)
            patterns["jwt_token"] = r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
            # Database Connection Strings with Passwords
            patterns["db_connection_uri"] = r"(?i)(?:postgres|mysql|mongodb|redis):\/\/[a-zA-Z0-9_-]+:[^@\s]+@[a-zA-Z0-9.-]+"

        if self.detect_pii:
            # US Social Security Number (SSN)
            patterns["ssn"] = r"\b\d{3}-\d{2}-\d{4}\b"
            # Credit Card Numbers (Major formats: Visa, MasterCard, Amex, Discover)
            patterns["credit_card"] = (
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"           # Visa
                r"5[1-5][0-9]{14}|"                         # MasterCard
                r"3[47][0-9]{13}|"                          # American Express
                r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"          # Diners Club
                r"6(?:011|5[0-9]{2})[0-9]{12})\b"           # Discover
            )
            # Email addresses
            patterns["email"] = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
            # Phone numbers (US / E.164-ish formats)
            patterns["phone_number"] = r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b"

        if self.detect_code_secrets:
            # Internal corporate domains & IP ranges
            patterns["internal_ip"] = r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3})\b"
            patterns["internal_hostname"] = r"(?i)\b[a-zA-Z0-9_\-]+\.(?:corp|internal|intranet|local|lan|priv)\b"
            # Confidentiality docstrings / corporate warnings
            patterns["confidential_notice"] = (
                r"(?i)\b(?:CONFIDENTIAL\s+AND\s+PROPRIETARY|"
                r"DO\s+NOT\s+DISTRIBUTE|"
                r"INTERNAL\s+USE\s+ONLY|"
                r"@internal\b|"
                r"TRADE\s+SECRET)\b"
            )

        self._patterns = {k: re.compile(v) for k, v in patterns.items()}

    def scan_text(self, text: str) -> PIIDetectionResult:
        """Scan text and return detailed detection result."""
        if not text:
            return PIIDetectionResult()

        detected_categories: Set[str] = set()
        total_matches = 0

        for category, pattern in self._patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_categories.add(category)
                total_matches += len(matches)

        has_sensitive = len(detected_categories) > 0
        requires_local = has_sensitive if self.enforce_local_only else False
        reason = ""
        if has_sensitive:
            categories_str = ", ".join(sorted(detected_categories))
            reason = f"Detected sensitive patterns: [{categories_str}] ({total_matches} matches)"

        return PIIDetectionResult(
            has_sensitive_data=has_sensitive,
            categories=sorted(list(detected_categories)),
            match_count=total_matches,
            requires_local_routing=requires_local,
            reason=reason,
        )

    def scan_messages(self, messages: List[Dict[str, Any]]) -> PIIDetectionResult:
        """Scan a list of OpenAI-style chat messages."""
        if not messages:
            return PIIDetectionResult()

        combined_text = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                combined_text.append(content)
            elif isinstance(content, list):
                # Multimodal or parts list
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        combined_text.append(part.get("text", ""))

        return self.scan_text("\n".join(combined_text))

    def mask_text(self, text: str) -> str:
        """Redact sensitive occurrences in text for safe logging/metrics."""
        masked = text
        for category, pattern in self._patterns.items():
            masked = pattern.sub(f"[REDACTED_{category.upper()}]", masked)
        return masked
