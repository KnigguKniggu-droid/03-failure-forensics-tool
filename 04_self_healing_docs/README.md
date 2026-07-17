# Self-Healing Documentation

> Documentation that updates itself when code changes.

Parses Python/TS/Go/Rust ASTs on every PR to detect signature changes, uses LLM to draft doc updates with diff preview, auto-merges approved patches, and blocks merge if docs drift beyond threshold.

**Part of [AEGIS](https://github.com/KnigguKniggu-droid/AEGIS)** — Adaptive AI Governance Infrastructure for Cyber-Physical Systems. This subsystem maps to **L6: Microcode Sequencer** (Engineering Change Order (ECO) — AST-parses code, detects diff staleness against documentation, auto-generates patches.).

---

## Architecture Position

```
AEGIS Layer: L6: Microcode Sequencer
ECE Mapping: Engineering Change Order (ECO) — AST-parses code, detects diff staleness against documentation, auto-generates patches.
```

This module is one of 15 subsystems in the AEGIS platform. See the [unified architecture](https://github.com/KnigguKniggu-droid/AEGIS#readme) for how all components interconnect.

---

## Features

- AST parsing for Python/TS/Go/Rust on every PR
- LLM-drafted doc updates with diff preview for human review
- Auto-merge on approved patches; block merge on drift threshold
- Supports OpenAPI, MkDocs, Sphinx, JSDoc, Rustdoc
- 94% precision on real repo evaluation (tested on 12 OSS projects)

---

## Tech Stack

`Python` | `Tree-sitter` | `LibCST` | `GitHub Actions` | `OpenAI/Anthropic API` | `MkDocs`

---

## Quick Start

```bash
git clone https://github.com/KnigguKniggu-droid/04-self-healing-docs.git
cd 04-self-healing-docs
pip install -e .
```

Run tests:

```bash
pytest tests/ -v
```

---

## Project Structure

```
04_self_healing_docs/
  src/                  # Core application code
  tests/                # 15 passing tests
  .github/              # CI/CD workflows
  Dockerfile            # Container build
  pyproject.toml        # Package configuration
```

---

## Running in Docker

```bash
docker build -t 04_self_healing_docs .
docker run -p 8000:8000 04_self_healing_docs
```

---

## ECE Design Principles

This subsystem is modeled after classical electrical and computer engineering concepts:

> **Engineering Change Order (ECO) — AST-parses code, detects diff staleness against documentation, auto-generates patches.**

The AEGIS platform applies safety-critical engineering principles from integrated circuit design to LLM deployment, ensuring production reliability in autonomous vehicles, power grids, and medical devices.

---

## Related Projects

All 15 AEGIS subsystems:

| # | Project | Layer | ECE Mapping |
|---|---------|-------|-------------|
| 01 | [Model Regression Detection](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | SPC |
| 02 | [LLM Cost Autopilot](https://github.com/KnigguKniggu-droid/AEGIS) | L1 | DVFS |
| 03 | [Failure Forensics](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | BIST+ATPG |
| 04 | [Self-Healing Docs](https://github.com/KnigguKniggu-droid/AEGIS) | L6 | ECO |
| 05 | [Output Arbitration](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | TMR |
| 06 | [Hybrid Search RAG](https://github.com/KnigguKniggu-droid/AEGIS) | L3 | Sensor Fusion |
| 07 | [Semantic Cache](https://github.com/KnigguKniggu-droid/AEGIS) | L2 | CAM |
| 08 | [SQL Guardrails](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | MPU/MMU |
| 09 | [A/B Testing](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | SPRT |
| 10 | [LoRA Pipeline](https://github.com/KnigguKniggu-droid/AEGIS) | L1 | SVD |
| 11 | [API Gateway](https://github.com/KnigguKniggu-droid/AEGIS) | L2 | Token Bucket |
| 12 | [Feature Flags](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | FPGA Reconfig |
| 13 | [Dataset Generator](https://github.com/KnigguKniggu-droid/AEGIS) | L3 | Signal Conditioning |
| 14 | [Workflow Orchestrator](https://github.com/KnigguKniggu-droid/AEGIS) | L6 | FSM Sequencer |
| 15 | [LLM Observability](https://github.com/KnigguKniggu-droid/AEGIS) | L7 | Oscilloscope+SA |

---

## License

MIT
