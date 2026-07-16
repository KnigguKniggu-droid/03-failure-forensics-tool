"""Typed contracts for the agentic workflow orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CHECKPOINTED = "checkpointed"


class AgentCapability(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CODE_GENERATION = "code_generation"
    REVIEW = "review"
    SUMMARIZATION = "summarization"
    VALIDATION = "validation"
    PLANNING = "planning"
    EXECUTION = "execution"


class TaskPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Task(BaseModel):
    """A unit of work to be dispatched to an agent."""

    task_id: str
    name: str
    description: str = ""
    required_capabilities: list[AgentCapability] = Field(..., min_length=1)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    assigned_agent_id: str | None = None
    parent_task_id: str | None = None
    child_task_ids: list[str] = Field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_seconds: int = 300
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentProfile(BaseModel):
    """Profile of an agent available for task dispatch."""

    agent_id: str
    name: str
    capabilities: list[AgentCapability] = Field(..., min_length=1)
    model: str = "gpt-4o"
    max_concurrent_tasks: int = 5
    current_load: int = 0
    success_rate: float = Field(1.0, ge=0.0, le=1.0)
    avg_latency_ms: float = 0.0
    is_available: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """A state checkpoint for workflow recovery."""

    checkpoint_id: str
    workflow_id: str
    task_id: str
    agent_id: str
    task_status: TaskStatus
    task_data: dict[str, Any] = Field(default_factory=dict)
    agent_state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = Field(..., ge=0)


class WorkflowState(BaseModel):
    """Complete state of a multi-agent workflow."""

    workflow_id: str
    name: str
    tasks: dict[str, Task] = Field(default_factory=dict)
    agents: dict[str, AgentProfile] = Field(default_factory=dict)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    current_checkpoint_seq: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task

    def add_agent(self, agent: AgentProfile) -> None:
        self.agents[agent.agent_id] = agent

    @property
    def pending_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    @property
    def running_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]

    @property
    def completed_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]

    @property
    def is_complete(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED) for t in self.tasks.values())


class WorkflowResult(BaseModel):
    """Final result of a completed workflow."""

    workflow_id: str
    status: TaskStatus
    completed_tasks: int
    failed_tasks: int
    total_duration_seconds: float
    task_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    checkpoint_count: int = 0
    final_output: dict[str, Any] = Field(default_factory=dict)
