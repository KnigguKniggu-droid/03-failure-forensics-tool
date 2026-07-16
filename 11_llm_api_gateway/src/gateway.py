"""Main gateway proxy with multi-vendor routing and fallback logic.

Routes traffic across multiple vendors, enforces rate limiting via
Redis token bucket, and falls back to alternate vendors when circuit
breakers open during service latency spikes.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

import httpx

from src.circuit_breaker import CircuitBreaker
from src.models import (
    CircuitState,
    GatewayConfig,
    GatewayRequest,
    GatewayResponse,
    VendorConfig,
)
from src.token_bucket import TokenBucketRateLimiter


class LLMGateway:
    """Enterprise LLM gateway with rate limiting and fallback routing."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.vendors: dict[str, VendorConfig] = {v.vendor_id: v for v in config.vendors}
        self.sorted_vendors = sorted(config.vendors, key=lambda v: v.priority)
        self.rate_limiter = TokenBucketRateLimiter(config.redis_url)
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        for vendor in config.vendors:
            self.circuit_breakers[vendor.vendor_id] = CircuitBreaker(
                vendor_id=vendor.vendor_id,
                failure_threshold=vendor.circuit_failure_threshold,
                recovery_timeout_s=vendor.circuit_recovery_timeout_s,
            )
        self._latency_samples: dict[str, list[float]] = {v.vendor_id: [] for v in config.vendors}

    async def route_request(self, request: GatewayRequest) -> GatewayResponse:
        """Route a request through the gateway with fallback support."""
        request_id = str(uuid.uuid4())

        vendor_order = self._get_vendor_order(request)

        for vendor in vendor_order:
            cb = self.circuit_breakers[vendor.vendor_id]
            if not cb.can_request():
                continue

            rl = self.rate_limiter.check_rate_limit(
                vendor.vendor_id,
                capacity=vendor.rate_limit_burst,
                refill_rate=vendor.rate_limit_rps,
            )
            if not rl.allowed:
                continue

            start = time.monotonic()
            try:
                result = await self._execute_vendor(vendor, request)
                latency = (time.monotonic() - start) * 1000

                self._record_latency(vendor.vendor_id, latency)
                cb.record_success(vendor.vendor_id)

                return GatewayResponse(
                    request_id=request_id,
                    vendor_id=vendor.vendor_id,
                    model=request.model,
                    content=result["content"],
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    latency_ms=latency,
                    fallback_used=vendor.vendor_id != vendor_order[0].vendor_id,
                    circuit_state=cb.get_state().state,
                )
            except Exception as exc:
                latency = (time.monotonic() - start) * 1000
                self._record_latency(vendor.vendor_id, latency)
                cb.record_failure(vendor.vendor_id)
                continue

        return GatewayResponse(
            request_id=request_id,
            vendor_id="none",
            model=request.model,
            content="",
            error="All vendors exhausted or circuit breakers open",
            circuit_state=CircuitState.OPEN,
        )

    def _get_vendor_order(self, request: GatewayRequest) -> list[VendorConfig]:
        """Determine the vendor routing order."""
        if request.preferred_vendor and request.preferred_vendor in self.vendors:
            preferred = self.vendors[request.preferred_vendor]
            rest = [v for v in self.sorted_vendors if v.vendor_id != preferred.vendor_id]
            return [preferred] + rest
        return list(self.sorted_vendors)

    async def _execute_vendor(
        self,
        vendor: VendorConfig,
        request: GatewayRequest,
    ) -> dict[str, Any]:
        """Execute a request against a specific vendor."""
        api_key = os.environ.get(vendor.api_key_env, "") if vendor.api_key_env else ""
        model = vendor.model_mapping.get(request.model, request.model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        timeout = vendor.timeout_ms / 1000.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{vendor.api_base}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        }

    def _record_latency(self, vendor_id: str, latency_ms: float) -> None:
        """Record a latency sample for rolling window metrics."""
        samples = self._latency_samples.get(vendor_id, [])
        samples.append(latency_ms)
        if len(samples) > 1000:
            samples = samples[-1000:]
        self._latency_samples[vendor_id] = samples

    def get_latency_metrics(self) -> dict[str, dict[str, float]]:
        """Get rolling window latency percentiles for all vendors."""
        import numpy as np
        metrics: dict[str, dict[str, float]] = {}
        for vendor_id, samples in self._latency_samples.items():
            if not samples:
                metrics[vendor_id] = {"p50": 0, "p95": 0, "p99": 0, "count": 0}
                continue
            arr = np.array(samples)
            metrics[vendor_id] = {
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "count": len(samples),
            }
        return metrics
