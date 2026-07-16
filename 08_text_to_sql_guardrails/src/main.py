"""FastAPI application for the Text-to-SQL guardrails system."""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.guardrails import validate_sql, validate_read_only
from src.models import GuardrailResult, TextToSQLRequest, TextToSQLResponse
from src.schema_introspector import SchemaIntrospector
from src.sql_executor import SQLExecutor

app = FastAPI(
    title="Text-to-SQL with Guardrails",
    description="SQLAlchemy introspection + sqlparse guardrails + hallucination detection",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_db_url = os.environ.get("DATABASE_URL", "duckdb:///data/analytics.duckdb")
_introspector = SchemaIntrospector(_db_url)
_executor = SQLExecutor(_db_url)


@app.get("/v1/schema")
async def get_schema() -> dict[str, Any]:
    """Return the introspected database schema."""
    schema = _introspector.introspect()
    return {
        "database": schema.database_name,
        "tables": [
            {
                "name": t.table_name,
                "columns": [c.model_dump() for c in t.columns],
            }
            for t in schema.tables
        ],
    }


@app.post("/v1/validate")
async def validate_query(sql: str) -> GuardrailResult:
    """Validate a SQL query against guardrails without executing it."""
    return validate_sql(sql)


@app.post("/v1/execute")
async def execute_query(sql: str, max_rows: int = 100) -> dict[str, Any]:
    """Execute a validated SQL query with guardrails."""
    guardrail = validate_read_only(sql)
    if not guardrail.is_safe:
        raise HTTPException(status_code=403, detail=f"Guardrail violation: {guardrail.rejected_reason}")
    result = _executor.execute(sql, max_rows=max_rows)
    return result.model_dump()


@app.post("/v1/cross-check")
async def cross_check_query(query_a: str, query_b: str, max_rows: int = 100) -> dict[str, Any]:
    """Run dual multi-query cross-checking for hallucination detection."""
    result = _executor.cross_check(query_a, query_b, max_rows=max_rows)
    return result.model_dump()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
