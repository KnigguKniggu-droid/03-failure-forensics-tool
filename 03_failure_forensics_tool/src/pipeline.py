"""4-step processing pipeline with OpenTelemetry context-managed spans.

Each step (Intake -> Extraction -> Classification -> Summarization) is
wrapped in a context-managed decorator that injects metrics into a
standard SpanData object. The pipeline builds a complete TraceRecord
that the forensic analyzer can inspect.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar

from src.models import (
    PipelineInput,
    PipelineOutput,
    PipelineStep,
    SpanData,
    SpanStatus,
    TraceRecord,
)

F = TypeVar("F", bound=Callable[..., Any])


class SpanContext:
    """Thread-local span context that tracks the current active span."""

    _current: SpanData | None = None
    _trace: TraceRecord | None = None

    @classmethod
    def set_trace(cls, trace: TraceRecord) -> None:
        cls._trace = trace

    @classmethod
    def get_trace(cls) -> TraceRecord | None:
        return cls._trace

    @classmethod
    def set_current_span(cls, span: SpanData | None) -> None:
        cls._current = span

    @classmethod
    def get_current_span(cls) -> SpanData | None:
        return cls._current

    @classmethod
    def reset(cls) -> None:
        cls._current = None
        cls._trace = None


@contextmanager
def span_context(step: PipelineStep, **span_kwargs: Any):
    """Context manager that creates a span, tracks metrics, and records it.

    Usage:
        with span_context(PipelineStep.EXTRACTION, input_tokens=100) as span:
            result = do_work()
            span.output_data = {"entities": result}
            span.model_confidence = 4
    """
    trace = SpanContext.get_trace()
    parent = SpanContext.get_current_span()

    span = SpanData(
        span_id=str(uuid.uuid4()),
        trace_id=trace.trace_id if trace else str(uuid.uuid4()),
        step=step,
        latency_ms=0.0,
        model_confidence=span_kwargs.get("model_confidence", 3),
        input_tokens=span_kwargs.get("input_tokens", 0),
        parent_span_id=parent.span_id if parent else None,
    )

    SpanContext.set_current_span(span)
    start = time.monotonic()
    try:
        yield span
        span.latency_ms = (time.monotonic() - start) * 1000
        if span.status == SpanStatus.OK and span.error_message is None:
            span.status = SpanStatus.OK
    except Exception as exc:
        span.latency_ms = (time.monotonic() - start) * 1000
        span.status = SpanStatus.ERROR
        span.error_message = str(exc)
        raise
    finally:
        if trace:
            trace.add_span(span)
        SpanContext.set_current_span(parent)


def traced_step(step: PipelineStep) -> Callable[[F], F]:
    """Decorator that wraps a pipeline step in a span context.

    The wrapped function must accept a PipelineInput or step data and
    return a dict with 'output', 'confidence', 'input_tokens', and
    'output_tokens' keys.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = SpanContext.get_trace()
            parent = SpanContext.get_current_span()

            span = SpanData(
                span_id=str(uuid.uuid4()),
                trace_id=trace.trace_id if trace else str(uuid.uuid4()),
                step=step,
                latency_ms=0.0,
                model_confidence=3,
                parent_span_id=parent.span_id if parent else None,
            )

            SpanContext.set_current_span(span)
            start = time.monotonic()
            try:
                result = func(*args, **kwargs)
                span.latency_ms = (time.monotonic() - start) * 1000
                if isinstance(result, dict):
                    span.output_data = result.get("output", {})
                    span.model_confidence = result.get("confidence", 3)
                    span.input_tokens = result.get("input_tokens", 0)
                    span.output_tokens = result.get("output_tokens", 0)
                    span.status = SpanStatus(result.get("status", SpanStatus.OK.value))
                if trace:
                    trace.add_span(span)
                return result
            except Exception as exc:
                span.latency_ms = (time.monotonic() - start) * 1000
                span.status = SpanStatus.ERROR
                span.error_message = str(exc)
                if trace:
                    trace.add_span(span)
                raise
            finally:
                SpanContext.set_current_span(parent)

        return wrapper  # type: ignore

    return decorator


class ForensicPipeline:
    """The 4-step processing pipeline with full tracing."""

    def __init__(self) -> None:
        self.trace = TraceRecord(trace_id=str(uuid.uuid4()))

    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        SpanContext.set_trace(self.trace)
        SpanContext.set_current_span(None)

        try:
            intake_result = self._intake(pipeline_input)
            extraction_result = self._extract(intake_result)
            classification_result = self._classify(extraction_result)
            summary_result = self._summarize(classification_result)

            self.trace.completed_at = datetime.now(timezone.utc)
            self.trace.final_output = summary_result.get("output", {}).get("summary", "")
            self.trace.final_status = SpanStatus.OK

            confidence_scores = {
                s.step.value: s.model_confidence
                for s in self.trace.spans
            }

            return PipelineOutput(
                trace_id=self.trace.trace_id,
                extracted_entities=extraction_result.get("output", {}),
                classification=classification_result.get("output", {}).get("label", ""),
                summary=summary_result.get("output", {}).get("summary", ""),
                confidence_scores=confidence_scores,
                spans=list(self.trace.spans),
            )
        except Exception:
            self.trace.final_status = SpanStatus.ERROR
            raise
        finally:
            SpanContext.reset()

    @traced_step(PipelineStep.INTAKE)
    def _intake(self, pipeline_input: PipelineInput) -> dict[str, Any]:
        raw = pipeline_input.raw_text
        tokens = max(1, len(raw) // 4)
        return {
            "output": {"raw_text": raw, "char_count": len(raw), "metadata": pipeline_input.metadata},
            "confidence": 5,
            "input_tokens": 0,
            "output_tokens": tokens,
            "status": "ok",
        }

    @traced_step(PipelineStep.EXTRACTION)
    def _extract(self, intake_result: dict[str, Any]) -> dict[str, Any]:
        raw = intake_result["output"]["raw_text"]
        tokens_in = max(1, len(raw) // 4)
        entities: dict[str, Any] = {
            "keywords": [w for w in raw.split() if len(w) > 4][:10],
            "entities": [],
            "key_phrases": raw.split(". ")[:3],
        }
        confidence = 4 if len(entities["keywords"]) > 3 else 2
        return {
            "output": entities,
            "confidence": confidence,
            "input_tokens": tokens_in,
            "output_tokens": max(1, len(str(entities)) // 4),
            "status": "ok",
        }

    @traced_step(PipelineStep.CLASSIFICATION)
    def _classify(self, extraction_result: dict[str, Any]) -> dict[str, Any]:
        entities = extraction_result["output"]
        keywords = entities.get("keywords", [])
        if any(k in ("error", "bug", "fail", "crash") for k in keywords):
            label = "incident"
        elif any(k in ("request", "feature", "enhance") for k in keywords):
            label = "feature_request"
        else:
            label = "general"
        confidence = 4 if len(keywords) > 5 else 3
        return {
            "output": {"label": label, "keyword_count": len(keywords)},
            "confidence": confidence,
            "input_tokens": max(1, len(str(entities)) // 4),
            "output_tokens": 10,
            "status": "ok",
        }

    @traced_step(PipelineStep.SUMMARIZATION)
    def _summarize(self, classification_result: dict[str, Any]) -> dict[str, Any]:
        label = classification_result["output"]["label"]
        summary = f"Classified as {label}. See extraction for details."
        return {
            "output": {"summary": summary},
            "confidence": 4,
            "input_tokens": max(1, len(str(classification_result)) // 4),
            "output_tokens": max(1, len(summary) // 4),
            "status": "ok",
        }


from datetime import datetime, timezone  # noqa: E402
