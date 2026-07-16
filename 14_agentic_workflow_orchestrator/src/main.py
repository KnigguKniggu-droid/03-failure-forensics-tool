"""FastAPI application for the Agentic Workflow Orchestrator."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Awaitable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.checkpoint import CheckpointManager
from src.dispatcher import TaskDispatcher
from src.models import (
    AgentCapability,
    AgentProfile,
    Task,
    TaskPriority,
    TaskStatus,
    WorkflowResult,
    WorkflowState,
)

app = FastAPI(
    title="Agentic Workflow Orchestrator",
    description="Multi-agent task dispatcher with state checkpoints",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_checkpoint_mgr = CheckpointManager()
_workflows: dict[str, WorkflowState] = {}
_dispatchers: dict[str, TaskDispatcher] = {}


class CreateWorkflowRequest(BaseModel):
    name: str
    agent_configs: list[dict[str, Any]] = []
    task_configs: list[dict[str, Any]] = []


@app.post("/v1/workflows")
async def create_workflow(request: CreateWorkflowRequest) -> dict[str, Any]:
    """Create a new workflow with agents and tasks."""
    workflow_id = str(uuid.uuid4())
    state = WorkflowState(workflow_id=workflow_id, name=request.name)

    for ac in request.agent_configs:
        agent = AgentProfile(
            agent_id=ac.get("agent_id", str(uuid.uuid4())),
            name=ac.get("name", "agent"),
            capabilities=[AgentCapability(c) for c in ac.get("capabilities", ["execution"])],
            model=ac.get("model", "gpt-4o"),
            max_concurrent_tasks=ac.get("max_concurrent_tasks", 5),
        )
        state.add_agent(agent)

    for tc in request.task_configs:
        task = Task(
            task_id=tc.get("task_id", str(uuid.uuid4())),
            name=tc.get("name", "task"),
            description=tc.get("description", ""),
            required_capabilities=[AgentCapability(c) for c in tc.get("required_capabilities", ["execution"])],
            priority=TaskPriority(tc.get("priority", 2)),
            input_data=tc.get("input_data", {}),
        )
        state.add_task(task)

    dispatcher = TaskDispatcher(state)
    _workflows[workflow_id] = state
    _dispatchers[workflow_id] = dispatcher
    _checkpoint_mgr.save_workflow_state(state)

    return {"workflow_id": workflow_id, "agents": len(state.agents), "tasks": len(state.tasks)}


@app.post("/v1/workflows/{workflow_id}/run")
async def run_workflow(workflow_id: str) -> dict[str, Any]:
    """Execute a workflow to completion."""
    dispatcher = _dispatchers.get(workflow_id)
    if dispatcher is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    result = await dispatcher.run_workflow()
    return result.model_dump()


@app.post("/v1/workflows/{workflow_id}/checkpoint")
async def create_checkpoint(workflow_id: str, task_id: str, agent_id: str) -> dict[str, Any]:
    """Create a checkpoint for the current workflow state."""
    state = _workflows.get(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    checkpoint = _checkpoint_mgr.create_checkpoint(state, task_id, agent_id)
    return {"checkpoint_id": checkpoint.checkpoint_id, "sequence": checkpoint.sequence_number}


@app.post("/v1/workflows/{workflow_id}/restore")
async def restore_workflow(workflow_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    """Restore a workflow from a checkpoint."""
    state = _checkpoint_mgr.restore_from_checkpoint(workflow_id, checkpoint_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow or checkpoint not found")
    _workflows[workflow_id] = state
    _dispatchers[workflow_id] = TaskDispatcher(state)
    return {"restored": True, "pending_tasks": len(state.pending_tasks)}


@app.get("/v1/workflows/{workflow_id}/status")
async def workflow_status(workflow_id: str) -> dict[str, Any]:
    """Get the current status of a workflow."""
    state = _workflows.get(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {
        "status": state.status.value,
        "total_tasks": len(state.tasks),
        "pending": len(state.pending_tasks),
        "running": len(state.running_tasks),
        "completed": len(state.completed_tasks),
        "checkpoints": len(state.checkpoints),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
