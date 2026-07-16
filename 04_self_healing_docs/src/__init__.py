"""Self-Healing Technical Documentation.

An AST repository parser that slices project code into semantic tokens,
links them to Markdown documentation blocks using cosine similarity,
detects staleness via git diff analysis, and produces structural edit
patches using an LLM reconciliation script.
"""

__version__ = "0.1.0"
