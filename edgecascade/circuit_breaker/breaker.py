"""Asynchronous Circuit Breaker state machine for model providers."""

from __future__ import annotations

import asyncio
import enum
import time
from typing import Dict, Optional


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"        # Normal, provider healthy
    OPEN = "OPEN"            # Tripped, traffic blocked / cascading
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """State machine preventing cascading failure during hardware saturation or provider outages."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_trials: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.half_open_max_trials = half_open_max_trials

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._successful_trials: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._last_state_change: float = time.time()
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def can_execute(self) -> bool:
        """Check if request is permitted through the breaker."""
        async with self._lock:
            now = time.time()
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if now - self._last_failure_time >= self.recovery_timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._successful_trials = 0
                    self._last_state_change = now
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                return self._successful_trials < self.half_open_max_trials

            return False

    async def record_success(self) -> None:
        """Record successful invocation."""
        async with self._lock:
            self._total_successes += 1
            now = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._successful_trials += 1
                if self._successful_trials >= self.half_open_max_trials:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._successful_trials = 0
                    self._last_state_change = now
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    async def record_failure(self, reason: Optional[str] = None) -> None:
        """Record failed invocation or timeout."""
        async with self._lock:
            self._total_failures += 1
            self._consecutive_failures += 1
            now = time.time()
            self._last_failure_time = now

            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if self._consecutive_failures >= self.failure_threshold or self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.OPEN
                    self._last_state_change = now

    async def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._successful_trials = 0
            self._last_state_change = time.time()

    def get_status(self) -> Dict[str, Any]:
        """Return status dictionary for observability metrics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "last_failure_ago_secs": (
                round(time.time() - self._last_failure_time, 2)
                if self._last_failure_time > 0
                else None
            ),
        }
