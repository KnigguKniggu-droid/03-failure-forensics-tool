"""SQLite persistence layer for regression baselines and run history.

Stores prompt configurations, baseline accuracy snapshots, and individual
run reports so that regression deltas can be computed across time.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.models import PromptConfig, RegressionReport

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "regression.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS prompt_configs (
            prompt_id    TEXT NOT NULL,
            version      TEXT NOT NULL,
            model        TEXT NOT NULL,
            config_json  TEXT NOT NULL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (prompt_id, version)
        );

        CREATE TABLE IF NOT EXISTS baselines (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id    TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            model        TEXT NOT NULL,
            accuracy     REAL NOT NULL,
            mean_judge_relevance REAL NOT NULL,
            total_items  INTEGER NOT NULL,
            recorded_at  TEXT NOT NULL DEFAULT (datetime('now')),
            report_json  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            prompt_id    TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            model        TEXT NOT NULL,
            accuracy     REAL NOT NULL,
            baseline_accuracy REAL NOT NULL,
            regression_delta REAL NOT NULL,
            severity     TEXT NOT NULL,
            mean_judge_relevance REAL NOT NULL,
            total_items  INTEGER NOT NULL,
            correct_count INTEGER NOT NULL,
            passed       INTEGER NOT NULL,
            blocks_merge INTEGER NOT NULL,
            timestamp    TEXT NOT NULL,
            report_json  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_runs_prompt ON runs(prompt_id, prompt_version);
        CREATE INDEX IF NOT EXISTS idx_baselines_prompt ON baselines(prompt_id, prompt_version);
        """
    )
    conn.commit()


def save_prompt_config(conn: sqlite3.Connection, config: PromptConfig) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO prompt_configs (prompt_id, version, model, config_json)
        VALUES (?, ?, ?, ?)
        """,
        (config.prompt_id, config.version, config.model, config.model_dump_json()),
    )
    conn.commit()


def get_baseline(
    conn: sqlite3.Connection, prompt_id: str, prompt_version: str
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT accuracy, mean_judge_relevance, total_items
        FROM baselines
        WHERE prompt_id = ? AND prompt_version = ?
        ORDER BY recorded_at DESC
        LIMIT 1
        """,
        (prompt_id, prompt_version),
    ).fetchone()
    return dict(row) if row else None


def set_baseline(conn: sqlite3.Connection, report: RegressionReport) -> None:
    conn.execute(
        """
        INSERT INTO baselines (prompt_id, prompt_version, model, accuracy,
                               mean_judge_relevance, total_items, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report.prompt_id,
            report.prompt_version,
            report.model,
            report.accuracy,
            report.mean_judge_relevance,
            report.total_items,
            report.model_dump_json(),
        ),
    )
    conn.commit()


def save_run(conn: sqlite3.Connection, report: RegressionReport) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO runs
            (run_id, prompt_id, prompt_version, model, accuracy, baseline_accuracy,
             regression_delta, severity, mean_judge_relevance, total_items,
             correct_count, passed, blocks_merge, timestamp, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report.run_id,
            report.prompt_id,
            report.prompt_version,
            report.model,
            report.accuracy,
            report.baseline_accuracy,
            report.regression_delta,
            report.severity.value,
            report.mean_judge_relevance,
            report.total_items,
            report.correct_count,
            int(report.passed),
            int(report.blocks_merge),
            report.timestamp.isoformat(),
            report.model_dump_json(),
        ),
    )
    conn.commit()


def load_prompt_config(
    conn: sqlite3.Connection, prompt_id: str, version: str
) -> PromptConfig | None:
    row = conn.execute(
        "SELECT config_json FROM prompt_configs WHERE prompt_id = ? AND version = ?",
        (prompt_id, version),
    ).fetchone()
    if not row:
        return None
    return PromptConfig.model_validate_json(row["config_json"])
