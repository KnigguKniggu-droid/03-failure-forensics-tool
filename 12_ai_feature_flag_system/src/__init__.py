"""AI Feature Flag System with Gradual Rollout and Quality Monitoring.

Client SDK evaluating flag states locally via consistent hash schemas.
Background Celery pipeline querying downstream outputs via LLM-as-judge.
Progressive rollout with automatic advancement and rollback on error spikes.
"""

__version__ = "0.1.0"
