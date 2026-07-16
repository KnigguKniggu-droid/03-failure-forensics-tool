"""State checkpointing for workflow recovery.

Persists workflow state at key transition points so that failed workflows
can be resumed from the last checkpoint rather than restarting from scratch.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from src.models import Checkpoint, TaskStatus, WorkflowState

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"


class CheckpointManager:
    """Manages workflow state checkpoints with SQLite persistence."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                task_status TEXT NOT NULL,
                task_data TEXT NOT NULL,
                agent_state TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workflow_states (
                workflow_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                checkpoint_count INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow ON checkpoints(workflow_id, sequence_number);
            """
        )
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def create_checkpoint(
        self,
        state: WorkflowState,
        task_id: str,
        agent_id: str,
        task_data: dict[str, Any] | None = None,
        agent_state: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint for the current workflow state."""
        task = state.tasks.get(task_id)
        task_status = task.status if task else TaskStatus.PENDING

        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            workflow_id=state.workflow_id,
            task_id=task_id,
            agent_id=agent_id,
            task_status=task_status,
            task_data=task_data or (task.input_data if task else {}),
            agent_state=agent_state or {},
            sequence_number=state.current_checkpoint_seq,
        )

        state.current_checkpoint_seq += 1
        state.checkpoints.append(checkpoint)

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO checkpoints
                (checkpoint_id, workflow_id, task_id, agent_id, task_status,
                 task_data, agent_state, sequence_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.checkpoint_id,
                checkpoint.workflow_id,
                checkpoint.task_id,
                checkpoint.agent_id,
                checkpoint.task_status.value,
                json.dumps(checkpoint.task_data),
                json.dumps(checkpoint.agent_state),
                checkpoint.sequence_number,
                checkpoint.created_at.isoformat(),
            ),
        )
        conn.commit()

        self.save_workflow_state(state)
        return checkpoint

    def save_workflow_state(self, state: WorkflowState) -> None:
        """Persist the complete workflow state."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO workflow_states
                (workflow_id, state_json, checkpoint_count, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                state.workflow_id,
                state.model_dump_json(),
                len(state.checkpoints),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    def load_workflow_state(self, workflow_id: str) -> WorkflowState | None:
        """Load a workflow state from persistence."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT state_json FROM workflow_states WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if not row:
            return None
        return WorkflowState.model_validate_json(row["state_json"])

    def get_checkpoints(self, workflow_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a workflow, ordered by sequence."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM checkpoints
            WHERE workflow_id = ?
            ORDER BY sequence_number ASC
            """,
            (workflow_id,),
        ).fetchall()
        return [
            Checkpoint(
                checkpoint_id=row["checkpoint_id"],
                workflow_id=row["workflow_id"],
                task_id=row["task_id"],
                agent_id=row["agent_id"],
                task_status=TaskStatus(row["task_status"]),
                task_data=json.loads(row["task_data"]),
                agent_state=json.loads(row["agent_state"]),
                sequence_number=row["sequence_number"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def restore_from_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str | None = None,
    ) -> WorkflowState | None:
        """Restore a workflow state from a specific checkpoint.

        If no checkpoint_id is provided, restores from the latest checkpoint.
        """
        state = self.load_workflow_state(workflow_id)
        if state is None:
            return None

        checkpoints = self.get_checkpoints(workflow_id)
        if not checkpoints:
            return state

        target = checkpoints[-1] if checkpoint_id is None else next(
            (c for c in checkpoints if c.checkpoint_id == checkpoint_id), None
        )
        if target is None:
            return state

        task = state.tasks.get(target.task_id)
        if task:
            task.status = target.task_status
            task.input_data = target.task_data

        for tid, t in state.tasks.items():
            if t.status == TaskStatus.RUNNING:
                t.status = TaskStatus.PENDING
                t.assigned_agent_id = None

        return state

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


from datetime import datetime, timezone  # noqa: E402
