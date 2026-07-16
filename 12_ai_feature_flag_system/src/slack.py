"""Slack webhook integration for rollout notifications."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from src.models import RolloutDecision


async def send_rollout_notification(
    decision: RolloutDecision,
    webhook_url: str | None = None,
) -> bool:
    """Send a Slack notification about a rollout decision."""
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        return False

    color = {
        "advance": "good",
        "hold": "warning",
        "rollback": "danger",
        "complete": "good",
    }.get(decision.action, "#cccccc")

    payload: dict[str, Any] = {
        "attachments": [
            {
                "color": color,
                "title": f"Feature Flag Rollout: {decision.flag_id}",
                "fields": [
                    {"title": "Action", "value": decision.action.upper(), "short": True},
                    {"title": "Percentage", "value": f"{decision.current_percentage:.0%} -> {decision.new_percentage:.0%}", "short": True},
                    {"title": "Reason", "value": decision.reason, "short": False},
                ],
                "footer": "AI Feature Flag System",
                "ts": int(decision.timestamp.timestamp()),
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            return resp.status_code == 200
    except Exception:
        return False
