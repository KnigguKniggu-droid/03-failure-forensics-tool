"""SQLite persistence for traces and forensic reports."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.models import ForensicReport, TraceRecord

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "forensics.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            final_output TEXT,
            final_status TEXT NOT NULL,
            total_latency_ms REAL NOT NULL,
            total_tokens INTEGER NOT NULL,
            min_confidence INTEGER NOT NULL,
            trace_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS forensic_reports (
            report_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            findings_count INTEGER NOT NULL,
            overall_assessment TEXT,
            report_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
        );

        CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(final_status);
        CREATE INDEX IF NOT EXISTS idx_reports_trace ON forensic_reports(trace_id);
        CREATE INDEX IF NOT EXISTS idx_reports_severity ON forensic_reports(severity);
        """
    )
    conn.commit()


def save_trace(conn: sqlite3.Connection, trace: TraceRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO traces
            (trace_id, started_at, completed_at, final_output, final_status,
             total_latency_ms, total_tokens, min_confidence, trace_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace.trace_id,
            trace.started_at.isoformat(),
            trace.completed_at.isoformat() if trace.completed_at else None,
            trace.final_output,
            trace.final_status.value,
            trace.total_latency_ms,
            trace.total_tokens,
            trace.min_confidence,
            trace.model_dump_json(),
        ),
    )
    conn.commit()


def save_report(conn: sqlite3.Connection, report: ForensicReport) -> None:
    report_id = str(report.trace_id) + "_report"
    conn.execute(
        """
        INSERT OR REPLACE INTO forensic_reports
            (report_id, trace_id, severity, findings_count, overall_assessment,
             report_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report_id,
            report.trace_id,
            report.severity,
            len(report.findings),
            report.overall_assessment,
            report.model_dump_json(),
            report.timestamp.isoformat(),
        ),
    )
    conn.commit()


def get_recent_traces(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM traces ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_report_for_trace(conn: sqlite3.Connection, trace_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM forensic_reports WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    return dict(row) if row else None
