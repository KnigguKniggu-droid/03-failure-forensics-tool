"""Failure Forensics Tool for AI Pipelines.

A 4-step processing pipeline (Intake -> Extraction -> Classification ->
Summarization) wrapped in OpenTelemetry context-managed spans. Includes
a backward forensic analyzer that traces data propagation faults through
a structured failure taxonomy.
"""

__version__ = "0.1.0"
