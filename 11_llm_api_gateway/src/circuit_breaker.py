"""Circuit breaker for fallback routing.

Monitors vendor health and opens circuit breakers during service
latency spikes or failure cascades, routing traffic to fallback vendors.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from src.models import CircuitBreakerState, CircuitState

HALF_OPEN_SUCCESS_THRESHOLD = 3


class CircuitBreaker:
    """Per-vendor circuit breaker with failure tracking and recovery."""

    def __init__(
        self,
        vendor_id: str,
        failure_threshold: int = 5,
        recovery_timeout_s: int = 30,
    ) -> None:
        self.vendor_id = vendor_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self._states: dict[str, CircuitBreakerState] = {}

    def get_state(self, vendor_id: str | None = None) -> CircuitBreakerState:
        vid = vendor_id or self.vendor_id
        if vid not in self._states:
            self._states[vid] = CircuitBreakerState(
                vendor_id=vid,
                state=CircuitState.CLOSED,
                failure_threshold=self.failure_threshold,
            )
        return self._states[vid]

    def can_request(self, vendor_id: str | None = None) -> bool:
        """Check if a request can be sent to this vendor."""
        state = self.get_state(vendor_id)

        if state.state == CircuitState.CLOSED:
            return True

        if state.state == CircuitState.OPEN:
            if state.last_failure_time is not None:
                elapsed = (datetime.now(timezone.utc) - state.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout_s:
                    state.state = CircuitState.HALF_OPEN
                    state.recovery_started_at = datetime.now(timezone.utc)
                    state.consecutive_successes = 0
                    return True
            return False

        if state.state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self, vendor_id: str | None = None) -> None:
        """Record a successful request."""
        state = self.get_state(vendor_id)
        state.consecutive_successes += 1

        if state.state == CircuitState.HALF_OPEN:
            if state.consecutive_successes >= HALF_OPEN_SUCCESS_THRESHOLD:
                state.state = CircuitState.CLOSED
                state.failure_count = 0
                state.last_failure_time = None
                state.recovery_started_at = None

    def record_failure(self, vendor_id: str | None = None) -> None:
        """Record a failed request."""
        state = self.get_state(vendor_id)
        state.failure_count += 1
        state.last_failure_time = datetime.now(timezone.utc)

        if state.state == CircuitState.HALF_OPEN:
            state.state = CircuitState.OPEN
            state.consecutive_successes = 0
        elif state.failure_count >= state.failure_threshold:
            state.state = CircuitState.OPEN

    def get_all_states(self) -> list[CircuitBreakerState]:
        """Get circuit breaker states for all tracked vendors."""
        return list(self._states.values())
