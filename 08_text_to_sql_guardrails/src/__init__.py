"""Text-to-SQL Interface with Guardrails and Hallucination Detection.

SQLAlchemy introspection for schema discovery, sqlparse-based security
middleware blocking DDL/DML, isolated read-only SELECT enforcement,
and dual multi-query cross-checking for hallucination detection.
"""

__version__ = "0.1.0"
