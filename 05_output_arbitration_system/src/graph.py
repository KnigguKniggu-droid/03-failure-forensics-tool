"""LangGraph state graph for the arbitration pipeline.

Defines the state schema and the graph topology:
  Input -> Fan Out (parallel critics) -> Fan In -> Adjudicator -> Output
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.adjudicator import adjudicate
from src.models import (
    AdjudicationInput,
    AdjudicationResult,
    ArbitrationRequest,
    ArbitrationResponse,
    CriticOutput,
)
from src.critics import run_all_critics

try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


class ArbitrationState(TypedDict, total=False):
    """State object passed through the LangGraph nodes."""

    query: str
    response: str
    context: str
    critic_outputs: list[CriticOutput]
    adjudication: AdjudicationResult
    total_latency_ms: float


def critic_fan_out_node(state: ArbitrationState) -> ArbitrationState:
    """Node that fans out to all three critics in parallel."""
    import asyncio
    import time

    start = time.monotonic()
    outputs = asyncio.run(
        run_all_critics(state["query"], state["response"], state.get("context", ""))
    )
    latency = (time.monotonic() - start) * 1000
    state["critic_outputs"] = outputs
    state["total_latency_ms"] = latency
    return state


def adjudicator_node(state: ArbitrationState) -> ArbitrationState:
    """Node that runs the central adjudicator on critic outputs."""
    adj_input = AdjudicationInput(
        query=state["query"],
        response=state["response"],
        critic_outputs=state["critic_outputs"],
    )
    state["adjudication"] = adjudicate(adj_input)
    return state


def build_arbitration_graph():
    """Build and compile the LangGraph arbitration pipeline."""
    if not HAS_LANGGRAPH:
        raise RuntimeError("LangGraph is not installed. Run: pip install langgraph")

    graph = StateGraph(ArbitrationState)
    graph.add_node("critics", critic_fan_out_node)
    graph.add_node("adjudicator", adjudicator_node)

    graph.set_entry_point("critics")
    graph.add_edge("critics", "adjudicator")
    graph.add_edge("adjudicator", END)

    return graph.compile()


async def run_arbitration(request: ArbitrationRequest) -> ArbitrationResponse:
    """Execute the full arbitration pipeline."""
    import asyncio
    import time
    import uuid

    start = time.monotonic()
    critic_outputs = await run_all_critics(request.query, request.response, request.context)
    critic_latency = (time.monotonic() - start) * 1000

    adj_input = AdjudicationInput(
        query=request.query,
        response=request.response,
        critic_outputs=critic_outputs,
    )
    result = adjudicate(adj_input)
    total_latency = (time.monotonic() - start) * 1000

    return ArbitrationResponse(
        request_id=str(uuid.uuid4()),
        adjudication=result,
        critic_outputs=critic_outputs,
        total_latency_ms=total_latency,
    )
