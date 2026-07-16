"""Streamlit forensic analysis dashboard."""

from __future__ import annotations

import sys
import json

import streamlit as st

sys.stdout.reconfigure(encoding="utf-8")

from src.db import get_connection, get_recent_traces, get_report_for_trace
from src.forensics import analyze_trace
from src.models import TraceRecord
from src.pipeline import ForensicPipeline
from src.models import PipelineInput

st.set_page_config(page_title="AI Pipeline Forensics", page_icon=":mag:", layout="wide")

st.title("AI Pipeline Failure Forensics")
st.markdown("Trace data propagation faults through the 4-step pipeline.")

tab1, tab2, tab3 = st.tabs(["Run Pipeline", "Trace Explorer", "Forensic Reports"])

with tab1:
    st.subheader("Run New Pipeline Trace")
    raw_input = st.text_area("Input text for pipeline processing", height=150, value="")
    if st.button("Run Pipeline", type="primary") and raw_input.strip():
        pipeline = ForensicPipeline()
        result = pipeline.run(PipelineInput(raw_text=raw_input))
        st.success(f"Pipeline completed. Trace ID: {result.trace_id}")

        st.subheader("Pipeline Steps")
        for span in result.spans:
            status_color = "green" if span.status.value == "ok" else "red"
            st.markdown(
                f"**{span.step.value.upper()}** "
                f":{status_color}[{span.status.value}] "
                f"| Confidence: {span.model_confidence}/5 "
                f"| Latency: {span.latency_ms:.1f}ms "
                f"| Tokens: {span.input_tokens} in / {span.output_tokens} out"
            )

        st.subheader("Output")
        st.json({
            "classification": result.classification,
            "summary": result.summary,
            "confidence_scores": result.confidence_scores,
        })

        if st.button("Run Forensic Analysis"):
            report = analyze_trace(pipeline.trace)
            st.session_state["last_report"] = report
            st.session_state["last_trace"] = pipeline.trace

            conn = get_connection()
            from src.db import save_trace, save_report
            save_trace(conn, pipeline.trace)
            save_report(conn, report)
            conn.close()

            if report.findings:
                st.warning(f"Found {len(report.findings)} fault(s). Severity: {report.severity}")
                for f in report.findings:
                    st.error(f"[{f.fault_type.value}] {f.description} (confidence: {f.confidence:.0%})")
            else:
                st.info("No faults detected in this trace.")

with tab2:
    st.subheader("Recent Traces")
    conn = get_connection()
    traces = get_recent_traces(conn, limit=20)
    if traces:
        for t in traces:
            status_icon = "OK" if t["final_status"] == "ok" else "ERR"
            with st.expander(f"{status_icon} | {t['trace_id'][:12]}... | {t['total_latency_ms']:.0f}ms | conf={t['min_confidence']}/5"):
                trace_data = json.loads(t["trace_json"])
                st.json(trace_data)
    else:
        st.info("No traces recorded yet. Run a pipeline first.")
    conn.close()

with tab3:
    st.subheader("Forensic Reports")
    conn = get_connection()
    traces = get_recent_traces(conn, limit=20)
    if traces:
        selected = st.selectbox(
            "Select trace",
            options=[t["trace_id"] for t in traces],
            format_func=lambda tid: f"{tid[:12]}...",
        )
        if selected:
            report_data = get_report_for_trace(conn, selected)
            if report_data:
                report_json = json.loads(report_data["report_json"])
                severity = report_json.get("severity", "unknown")
                if severity == "critical":
                    st.error(f"Severity: {severity.upper()}")
                elif severity == "high":
                    st.warning(f"Severity: {severity.upper()}")
                else:
                    st.info(f"Severity: {severity.upper()}")

                st.text(report_json.get("overall_assessment", ""))

                findings = report_json.get("findings", [])
                if findings:
                    for f in findings:
                        st.json(f)
                else:
                    st.success("No faults detected.")
            else:
                st.info("No forensic report for this trace.")
    else:
        st.info("No traces available.")
    conn.close()
