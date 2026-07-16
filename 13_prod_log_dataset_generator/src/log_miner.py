"""Production log miner for ClickHouse and PostgreSQL.

Mines production LLM interaction logs from ClickHouse or PostgreSQL,
filters by time range and quality, and returns structured log entries.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from src.models import LogSource, MiningConfig, ProductionLog

CLICKHOUSE_QUERY = """
SELECT
    log_id,
    timestamp,
    user_query,
    model_response,
    model,
    input_tokens,
    output_tokens,
    latency_ms,
    success,
    error,
    user_feedback
FROM llm_interaction_logs
WHERE timestamp >= {start_time:DateTime}
  AND length(user_query) >= {min_length:UInt32}
  AND success = 1
ORDER BY timestamp DESC
LIMIT {max_logs:UInt32}
"""

POSTGRES_QUERY = """
SELECT log_id, timestamp, user_query, model_response, model,
       input_tokens, output_tokens, latency_ms, success, error, user_feedback
FROM llm_interaction_logs
WHERE timestamp >= %s
  AND length(user_query) >= %s
  AND success = true
ORDER BY timestamp DESC
LIMIT %s
"""


class LogMiner:
    """Mines production logs from ClickHouse or PostgreSQL."""

    def __init__(self, config: MiningConfig) -> None:
        self.config = config

    def mine_clickhouse(self) -> list[ProductionLog]:
        """Mine logs from ClickHouse."""
        try:
            import clickhouse_connect
            client = clickhouse_connect.get_client(
                host=self.config.connection_url or os.environ.get("CLICKHOUSE_HOST", "localhost"),
                port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
                username=os.environ.get("CLICKHOUSE_USER", "default"),
                password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            )
            start_time = datetime.now(timezone.utc) - timedelta(hours=self.config.time_range_hours)
            result = client.query(
                CLICKHOUSE_QUERY,
                parameters={
                    "start_time": start_time,
                    "min_length": self.config.min_query_length,
                    "max_logs": self.config.max_logs,
                },
            )
            return self._parse_rows(result.result_rows, result.column_names)
        except ImportError:
            return self._mock_mine()

    def mine_postgres(self) -> list[ProductionLog]:
        """Mine logs from PostgreSQL."""
        try:
            import psycopg2
            start_time = datetime.now(timezone.utc) - timedelta(hours=self.config.time_range_hours)
            conn = psycopg2.connect(self.config.connection_url)
            cursor = conn.cursor()
            cursor.execute(POSTGRES_QUERY, (start_time, self.config.min_query_length, self.config.max_logs))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            return self._parse_rows(rows, columns)
        except ImportError:
            return self._mock_mine()

    def mine(self) -> list[ProductionLog]:
        """Mine logs from the configured source."""
        if self.config.source == LogSource.CLICKHOUSE:
            return self.mine_clickhouse()
        elif self.config.source == LogSource.POSTGRES:
            return self.mine_postgres()
        return self._mock_mine()

    def _parse_rows(self, rows: list[tuple], columns: list[str]) -> list[ProductionLog]:
        """Parse database rows into ProductionLog objects."""
        logs: list[ProductionLog] = []
        for row in rows:
            data = dict(zip(columns, row))
            logs.append(ProductionLog(
                log_id=str(data.get("log_id", "")),
                timestamp=data.get("timestamp", datetime.now(timezone.utc)),
                user_query=data.get("user_query", ""),
                model_response=data.get("model_response", ""),
                model=data.get("model", ""),
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                latency_ms=float(data.get("latency_ms", 0)),
                success=bool(data.get("success", True)),
                error=data.get("error"),
                user_feedback=data.get("user_feedback"),
            ))
        return logs

    def _mock_mine(self) -> list[ProductionLog]:
        """Generate mock production logs for testing without a database."""
        import random
        import uuid

        templates = [
            ("How do I reset my password?", "account", "easy"),
            ("What is the API rate limit?", "technical", "easy"),
            ("Explain the billing cycle", "billing", "medium"),
            ("How to integrate with webhook", "technical", "medium"),
            ("Compare pricing plans", "billing", "medium"),
            ("Debug 500 error on /users", "technical", "hard"),
            ("Refund for duplicate charge", "billing", "easy"),
            ("GDPR data export request", "general", "medium"),
            ("Configure SSO for enterprise", "account", "hard"),
            ("Why is my query slow?", "technical", "hard"),
            ("Update payment method", "billing", "easy"),
            ("Add team member to workspace", "account", "medium"),
            ("Export analytics report", "general", "medium"),
            ("API authentication failing", "technical", "medium"),
            ("Cancel subscription", "billing", "easy"),
        ]

        logs: list[ProductionLog] = []
        for i in range(min(self.config.max_logs, 200)):
            query, category, difficulty = random.choice(templates)
            log = ProductionLog(
                log_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=random.randint(0, self.config.time_range_hours)),
                user_query=f"{query} (variation {i})",
                model_response=f"Here is the response to: {query}",
                model=random.choice(["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]),
                input_tokens=random.randint(10, 500),
                output_tokens=random.randint(50, 2000),
                latency_ms=random.uniform(100, 5000),
                success=random.random() > 0.05,
                metadata={"category": category, "difficulty": difficulty},
            )
            logs.append(log)
        return logs
