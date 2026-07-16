"""SQLAlchemy database introspection for schema discovery.

Fetches system table schemas using SQLAlchemy introspection to provide
context for natural language SQL generation.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from src.models import ColumnSchema, DatabaseSchema, TableSchema


class SchemaIntrospector:
    """Introspects a database via SQLAlchemy to extract table schemas."""

    def __init__(self, connection_url: str) -> None:
        self.connection_url = connection_url
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.connection_url)
        return self._engine

    def introspect(self) -> DatabaseSchema:
        """Extract all table schemas from the connected database."""
        inspector = inspect(self.engine)
        tables: list[TableSchema] = []

        for table_name in inspector.get_table_names():
            columns: list[ColumnSchema] = []
            pk_cols = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))

            for col in inspector.get_columns(table_name):
                fks = inspector.get_foreign_keys(table_name)
                fk_ref = None
                for fk in fks:
                    if col["name"] in fk.get("constrained_columns", []):
                        referred = fk.get("referred_columns", [])
                        referred_table = fk.get("referred_table", "")
                        if referred:
                            fk_ref = f"{referred_table}.{referred[0]}"
                        break

                columns.append(ColumnSchema(
                    name=col["name"],
                    type=str(col["type"]),
                    nullable=col.get("nullable", True),
                    primary_key=col["name"] in pk_cols,
                    foreign_key=fk_ref,
                    default=str(col.get("default", "")) if col.get("default") else None,
                ))

            tables.append(TableSchema(
                table_name=table_name,
                columns=columns,
                schema_name=inspector.default_schema_name or "public",
            ))

        db_name = self.engine.url.database or ""
        return DatabaseSchema(tables=tables, database_name=db_name)

    def get_table_schema(self, table_name: str) -> TableSchema | None:
        """Get schema for a single table."""
        schema = self.introspect()
        for t in schema.tables:
            if t.table_name == table_name:
                return t
        return None

    def format_schema_for_prompt(self, schema: DatabaseSchema | None = None) -> str:
        """Format the schema as a text prompt for SQL generation."""
        schema = schema or self.introspect()
        lines: list[str] = [f"Database: {schema.database_name}", ""]
        for table in schema.tables:
            lines.append(f"Table: {table.table_name}")
            for col in table.columns:
                pk = " PRIMARY KEY" if col.primary_key else ""
                fk = f" REFERENCES {col.foreign_key}" if col.foreign_key else ""
                nullable = "" if col.nullable else " NOT NULL"
                lines.append(f"  - {col.name} {col.type}{pk}{fk}{nullable}")
            lines.append("")
        return "\n".join(lines)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
