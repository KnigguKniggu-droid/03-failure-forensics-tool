"""Multi-agent task dispatcher with capability-based routing.

Routes tasks to the most suitable available agent based on required
capabilities, current load, and historical success rates.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Awaitable

from src.models import (
    AgentCapability,
    AgentProfile,
    Task,
    TaskPriority,
    TaskStatus,
    WorkflowResult,
    WorkflowState,
)


class TaskDispatcher:
    """Dispatches tasks to agents based on capability matching and load balancing."""

    def __init__(self, state: WorkflowState) -> None:
        self.state = state
        self._task_handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    def register_handler(
        self,
        capability: str,
        handler: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        """Register a task handler for a specific capability."""
        self._task_handlers[capability] = handler

    def select_agent(self, task: Task) -> AgentProfile | None:
        """Select the best available agent for a task.

        Selection criteria (in order):
        1. Agent has all required capabilities
        2. Agent is available
        3. Agent has the lowest current load
        4. Agent has the highest success rate (tiebreaker)
        """
        candidates = [
            agent for agent in self.state.agents.values()
            if agent.is_available
            and all(cap in agent.capabilities for cap in task.required_capabilities)
            and agent.current_load < agent.max_concurrent_tasks
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda a: (a.current_load, -a.success_rate, a.avg_latency_ms))
        return candidates[0]

    def get_next_task(self) -> Task | None:
        """Get the highest-priority pending task."""
        pending = self.state.pending_tasks
        if not pending:
            return None
        pending.sort(key=lambda t: (-t.priority.value, t.created_at))
        return pending[0]

    async def dispatch_task(self, task: Task) -> dict[str, Any]:
        """Dispatch a single task to an agent and execute it."""
        agent = self.select_agent(task)
        if agent is None:
            task.status = TaskStatus.PENDING
            return {"error": "no available agent", "task_id": task.task_id}

        task.assigned_agent_id = agent.agent_id
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        agent.current_load += 1

        try:
            primary_cap = task.required_capabilities[0]
            handler = self._task_handlers.get(primary_cap)

            if handler:
                result = await asyncio.wait_for(
                    handler(task=task, agent=agent),
                    timeout=task.timeout_seconds,
                )
            else:
                result = await self._default_handler(task, agent)

            task.output_data = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            return result

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.metadata["error"] = "timeout"
            return {"error": "timeout", "task_id": task.task_id}
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.metadata["error"] = str(exc)
            return {"error": str(exc), "task_id": task.task_id}
        finally:
            agent.current_load = max(0, agent.current_load - 1)

    async def _default_handler(self, task: Task, agent: AgentProfile) -> dict[str, Any]:
        """Default handler that simulates task execution."""
        await asyncio.sleep(0.01)
        return {
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "result": f"Processed by {agent.name}",
            "capabilities_used": [c.value for c in task.required_capabilities],
        }

    async def run_workflow(self) -> WorkflowResult:
        """Execute all tasks in the workflow until completion."""
        self.state.status = TaskStatus.RUNNING
        start = time.monotonic()

        while not self.state.is_complete:
            task = self.get_next_task()
            if task is None:
                await asyncio.sleep(0.1)
                continue

            await self.dispatch_task(task)

        duration = time.monotonic() - start
        self.state.completed_at = datetime.now(timezone.utc)
        self.state.status = TaskStatus.COMPLETED

        completed = len(self.state.completed_tasks)
        failed = sum(1 for t in self.state.tasks.values() if t.status == TaskStatus.FAILED)

        return WorkflowResult(
            workflow_id=self.state.workflow_id,
            status=self.state.status,
            completed_tasks=completed,
            failed_tasks=failed,
            total_duration_seconds=duration,
            task_results={tid: t.output_data for tid, t in self.state.tasks.items()},
            checkpoint_count=len(self.state.checkpoints),
            final_output={
                tid: t.output_data
                for tid, t in self.state.tasks.items()
                if t.status == TaskStatus.COMPLETED and t.parent_task_id is None
            },
        )


from datetime import datetime, timezone  # noqa: E402
