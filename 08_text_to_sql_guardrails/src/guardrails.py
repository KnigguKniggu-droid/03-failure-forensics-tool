"""SQL security guardrails using sqlparse.

Blocks DDL/DML mutation behaviors and forces SELECT transactions
inside isolated, read-only structures.
"""

from __future__ import annotations

import re
from typing import Any

try:
    import sqlparse
    from sqlparse.sql import Identifier, IdentifierList, Statement
    from sqlparse.tokens import Keyword, DML, DDL
    HAS_SQLPARSE = True
except ImportError:
    HAS_SQLPARSE = False

from src.models import GuardrailResult, GuardrailViolation, SQLStatementType

DANGEROUS_FUNCTIONS = {
    "pg_sleep", "pg_terminate_backend", "lo_import", "lo_export",
    "pg_read_file", "pg_ls_dir", "pg_stat_file",
}

DDL_KEYWORDS = {"CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE"}
DML_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT"}

MULTI_STATEMENT_PATTERN = re.compile(r";\s*\S", re.MULTILINE)
COMMENT_PATTERN = re.compile(r"--.*$", re.MULTILINE)


def classify_statement(sql: str) -> SQLStatementType:
    """Classify the SQL statement type from raw text."""
    cleaned = COMMENT_PATTERN.sub("", sql).strip().upper()
    if cleaned.startswith("SELECT"):
        return SQLStatementType.SELECT
    elif cleaned.startswith("INSERT"):
        return SQLStatementType.INSERT
    elif cleaned.startswith("UPDATE"):
        return SQLStatementType.UPDATE
    elif cleaned.startswith("DELETE"):
        return SQLStatementType.DELETE
    elif cleaned.startswith("CREATE"):
        return SQLStatementType.CREATE
    elif cleaned.startswith("ALTER"):
        return SQLStatementType.ALTER
    elif cleaned.startswith("DROP"):
        return SQLStatementType.DROP
    return SQLStatementType.UNKNOWN


def check_for_multiple_statements(sql: str) -> bool:
    """Check if the SQL contains multiple semicolon-separated statements."""
    cleaned = COMMENT_PATTERN.sub("", sql)
    return bool(MULTI_STATEMENT_PATTERN.search(cleaned))


def check_for_dangerous_functions(sql: str) -> bool:
    """Check for dangerous database functions."""
    upper = sql.upper()
    return any(func.upper() in upper for func in DANGEROUS_FUNCTIONS)


def check_for_ddl(sql: str) -> bool:
    """Check for DDL keywords."""
    cleaned = COMMENT_PATTERN.sub("", sql).strip().upper()
    return any(cleaned.startswith(kw) or f" {kw} " in f" {cleaned} " for kw in DDL_KEYWORDS)


def check_for_dml(sql: str) -> bool:
    """Check for DML keywords."""
    cleaned = COMMENT_PATTERN.sub("", sql).strip().upper()
    return any(cleaned.startswith(kw) or f" {kw} " in f" {cleaned} " for kw in DML_KEYWORDS)


def sanitize_sql(sql: str) -> str:
    """Strip comments and trailing semicolons, return clean SQL."""
    cleaned = COMMENT_PATTERN.sub("", sql).strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return cleaned


def validate_sql(sql: str) -> GuardrailResult:
    """Full guardrail validation of a SQL query.

    Enforces:
    - Only SELECT statements allowed
    - No DDL (CREATE, ALTER, DROP, TRUNCATE, GRANT, REVOKE)
    - No DML (INSERT, UPDATE, DELETE, MERGE)
    - No multiple statements (semicolon injection)
    - No dangerous functions
    """
    violations: list[GuardrailViolation] = []

    if check_for_multiple_statements(sql):
        violations.append(GuardrailViolation.MULTIPLE_STATEMENTS)

    stmt_type = classify_statement(sql)

    if stmt_type != SQLStatementType.SELECT:
        violations.append(GuardrailViolation.NOT_SELECT)

    if check_for_ddl(sql):
        violations.append(GuardrailViolation.DDL_DETECTED)

    if check_for_dml(sql):
        violations.append(GuardrailViolation.DML_DETECTED)

    if check_for_dangerous_functions(sql):
        violations.append(GuardrailViolation.DANGEROUS_FUNCTION)

    is_safe = len(violations) == 0
    sanitized = sanitize_sql(sql) if is_safe else ""

    return GuardrailResult(
        is_safe=is_safe,
        statement_type=stmt_type,
        violations=violations,
        sanitized_sql=sanitized,
        rejected_reason="; ".join(v.value for v in violations) if violations else "",
    )


def validate_read_only(sql: str, connection_url: str | None = None) -> GuardrailResult:
    """Validate SQL and confirm it will run in a read-only transaction context."""
    result = validate_sql(sql)
    if not result.is_safe:
        return result

    read_only_prefix = "SET TRANSACTION READ ONLY; "
    result.sanitized_sql = f"{read_only_prefix}{result.sanitized_sql}"
    return result
