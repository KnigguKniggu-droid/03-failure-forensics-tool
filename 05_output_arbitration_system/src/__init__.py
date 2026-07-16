"""LLM Output Arbitration System.

A LangGraph-based system that fans out to parallel critic nodes
(Factual Accuracy via GPT-4o, Logical Consistency via Claude,
Completeness via Local Llama) and aggregates outputs into an
evidence chain resolved by a central Adjudicator scoring 1 to 10.
"""

__version__ = "0.1.0"
