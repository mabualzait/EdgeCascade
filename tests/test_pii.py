"""Tests for Privacy, Secret, and PII Gate."""

import pytest
from edgecascade.pii.detector import PIIDetector


def test_pii_clean_prompt():
    detector = PIIDetector()
    result = detector.scan_text("Summarize the following article in three bullet points.")
    assert not result.has_sensitive_data
    assert not result.requires_local_routing
    assert result.match_count == 0


def test_pii_aws_secret_key():
    detector = PIIDetector()
    text = "My AWS key is AKIAIOSFODNN7EXAMPLE and secret is aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"
    result = detector.scan_text(text)
    assert result.has_sensitive_data
    assert result.requires_local_routing
    assert "aws_access_key" in result.categories
    assert "aws_secret_key" in result.categories


def test_pii_private_key():
    detector = PIIDetector()
    text = """
    Here is our certificate:
    -----BEGIN RSA PRIVATE KEY-----
    MIIEowIBAAKCAQEA0Y1+example==
    -----END RSA PRIVATE KEY-----
    """
    result = detector.scan_text(text)
    assert result.has_sensitive_data
    assert result.requires_local_routing
    assert "private_key" in result.categories


def test_pii_ssn_and_credit_card():
    detector = PIIDetector()
    text = "User SSN is 123-45-6789 and visa card is 4111111111111111"
    result = detector.scan_text(text)
    assert result.has_sensitive_data
    assert result.requires_local_routing
    assert "ssn" in result.categories
    assert "credit_card" in result.categories


def test_pii_internal_code_confidential():
    detector = PIIDetector()
    text = "// CONFIDENTIAL AND PROPRIETARY. Connect to 10.240.12.5 and database.internal.corp"
    result = detector.scan_text(text)
    assert result.has_sensitive_data
    assert result.requires_local_routing
    assert "internal_ip" in result.categories
    assert "internal_hostname" in result.categories
    assert "confidential_notice" in result.categories


def test_pii_masking():
    detector = PIIDetector()
    text = "Contact me at alice@company.com with SSN 987-65-4321."
    masked = detector.mask_text(text)
    assert "alice@company.com" not in masked
    assert "987-65-4321" not in masked
    assert "[REDACTED_EMAIL]" in masked
    assert "[REDACTED_SSN]" in masked
