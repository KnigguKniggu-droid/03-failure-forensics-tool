"""Typed contracts for the Text-to-SQL guardrails system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SQLStatementType(str, Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DROP = "drop"
    CREATE = "create"
    ALTER = "alter"
    UNKNOWN = "unknown"


class GuardrailViolation(str, Enum):
    DDL_DETECTED = "ddl_detected"
    DML_DETECTED = "dml_detected"
    NOT_SELECT = "not_select"
    MULTIPLE_STATEMENTS = "multiple_statements"
    DANGEROUS_FUNCTION = "dangerous_function"
    SUBQUERY_INJECTION = "subquery_injection"


class ColumnSchema(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None
    default: Any = None


class TableSchema(BaseModel):
    table_name: str
    columns: list[ColumnSchema] = Field(default_factory=list)
    row_count: int | None = None
    schema_name: str = "public"


class DatabaseSchema(BaseModel):
    tables: list[TableSchema] = Field(default_factory=list)
    database_name: str = ""
    introspected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GuardrailResult(BaseModel):
    """Result of guardrail validation on a SQL query."""

    is_safe: bool
    statement_type: SQLStatementType
    violations: list[GuardrailViolation] = Field(default_factory=list)
    sanitized_sql: str = ""
    rejected_reason: str = ""


class SQLExecutionResult(BaseModel):
    """Result of executing a validated SQL query."""

    query: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_ms: float = 0.0
    error: str | None = None


class CrossCheckResult(BaseModel):
    """Result of dual multi-query cross-checking for hallucination detection."""

    query_a: str
    query_b: str
    result_a: SQLExecutionResult
    result_b: SQLExecutionResult
    results_match: bool
    hallucination_flagged: bool
    discrepancy_details: str = ""


class TextToSQLRequest(BaseModel):
    """API request for natural language to SQL."""

    natural_language_query: str = Field(..., min_length=1)
    max_rows: int = Field(100, ge=1, le=10000)


class TextToSQLResponse(BaseModel):
    """API response with generated SQL, results, and guardrail status."""

    request_id: str
    natural_language_query: str
    generated_sql: str
    guardrail: GuardrailResult
    execution: SQLExecutionResult | None = None
    cross_check: CrossCheckResult | None = None
    schema_used: list[str] = Field(default_factory=list)
