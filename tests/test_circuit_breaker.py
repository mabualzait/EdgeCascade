"""Tests for Circuit Breaker."""

import asyncio
import pytest
from edgecascade.circuit_breaker.breaker import CircuitBreaker, CircuitState


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    cb = CircuitBreaker(
        name="test_cb",
        failure_threshold=2,
        recovery_timeout_seconds=0.05,
        half_open_max_trials=1,
    )

    # Initial state: CLOSED
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute()

    # 1 failure: Still CLOSED
    await cb.record_failure("error 1")
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute()

    # 2nd failure: Trips to OPEN
    await cb.record_failure("error 2")
    assert cb.state == CircuitState.OPEN
    assert not await cb.can_execute()

    # Wait for recovery timeout
    await asyncio.sleep(0.06)

    # State transitions to HALF_OPEN on first trial
    assert await cb.can_execute()
    assert cb.state == CircuitState.HALF_OPEN

    # Success closes the breaker back to normal
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute()
