"""Semantic Caching Layer for LLM APIs.

An OpenAI-compatible FastAPI middleware that routes vector queries to
Redis VL with a 0.95 similarity boundary. Prevents index bleed via
unique cache keys embedding system prompts, temperatures, and model
metadata hashes. Streams partial responses while buffering finalized
output into Redis.
"""

__version__ = "0.1.0"
