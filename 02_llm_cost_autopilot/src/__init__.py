"""LLM Cost Autopilot.

An intelligent multi-provider routing proxy that classifies request
complexity and routes to the most cost-effective model that can handle
the task. Includes a downstream verification evaluator that feeds routing
failures back into weekly model adaptation metrics.
"""

__version__ = "0.1.0"
