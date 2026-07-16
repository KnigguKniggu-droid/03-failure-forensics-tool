"""SQL execution engine with dual multi-query cross-checking.

Runs two distinct multi-query syntax configurations and drops flags
when results do not align, detecting potential SQL hallucination.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.guardrails import validate_sql
from src.models import CrossCheckResult, GuardrailResult, SQLExecutionResult


class SQLExecutor:
    """Executes validated SQL queries in an isolated read-only context."""

    def __init__(self, connection_url: str) -> None:
        self.connection_url = connection_url
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.connection_url)
        return self._engine

    def execute(self, sql: str, max_rows: int = 100) -> SQLExecutionResult:
        """Execute a single validated SQL query."""
        guardrail = validate_sql(sql)
        if not guardrail.is_safe:
            return SQLExecutionResult(
                query=sql,
                error=f"Guardrail violation: {guardrail.rejected_reason}",
            )

        start = time.monotonic()
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(guardrail.sanitized_sql))
                columns = list(result.keys()) if result.returns_rows else []
                rows = [dict(row._mapping) for row in result.fetchmany(max_rows)]
                latency = (time.monotonic() - start) * 1000
                return SQLExecutionResult(
                    query=sql,
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_ms=latency,
                )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            return SQLExecutionResult(
                query=sql,
                execution_ms=latency,
                error=str(exc),
            )

    def cross_check(
        self,
        query_a: str,
        query_b: str,
        max_rows: int = 100,
    ) -> CrossCheckResult:
        """Run two distinct SQL configurations and compare results.

        If the results do not align, a hallucination flag is raised.
        """
        result_a = self.execute(query_a, max_rows)
        result_b = self.execute(query_b, max_rows)

        match = _results_match(result_a, result_b)
        discrepancy = "" if match else _describe_discrepancy(result_a, result_b)

        return CrossCheckResult(
            query_a=query_a,
            query_b=query_b,
            result_a=result_a,
            result_b=result_b,
            results_match=match,
            hallucination_flagged=not match,
            discrepancy_details=discrepancy,
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


def _results_match(a: SQLExecutionResult, b: SQLExecutionResult) -> bool:
    """Check if two execution results produce the same data."""
    if a.error or b.error:
        return False
    if a.row_count != b.row_count:
        return False
    if set(a.columns) != set(b.columns):
        return False
    a_vals = sorted(tuple(sorted(row.items())) for row in a.rows)
    b_vals = sorted(tuple(sorted(row.items())) for row in b.rows)
    return a_vals == b_vals


def _describe_discrepancy(a: SQLExecutionResult, b: SQLExecutionResult) -> str:
    """Describe the discrepancy between two results."""
    parts: list[str] = []
    if a.error and b.error:
        return f"Both queries failed: A={a.error}, B={b.error}"
    if a.error:
        return f"Query A failed: {a.error}"
    if b.error:
        return f"Query B failed: {b.error}"
    if a.row_count != b.row_count:
        parts.append(f"Row count mismatch: A={a.row_count}, B={b.row_count}")
    if set(a.columns) != set(b.columns):
        parts.append(f"Column mismatch: A={set(a.columns)}, B={set(b.columns)}")
    if not parts:
        parts.append("Row data mismatch despite matching shape")
    return "; ".join(parts)
