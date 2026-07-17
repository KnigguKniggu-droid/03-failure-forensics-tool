# AI Feature Flag System

> Feature flags for LLM features with auto-rollback.

Boolean + multivariate flags for prompts/models, canary rollouts from 1% to 100%, LLM-as-judge evaluation on shadow traffic, auto-rollback on quality regression.

**Part of [AEGIS](https://github.com/KnigguKniggu-droid/AEGIS)** — Adaptive AI Governance Infrastructure for Cyber-Physical Systems. This subsystem maps to **L5: Adaptive Control** (Dynamic reconfiguration (FPGA) — canary rollout with 1->10->50->100% progression, auto-rollback on error threshold, identical to partial reconfiguration.).

---

## Architecture Position

```
AEGIS Layer: L5: Adaptive Control
ECE Mapping: Dynamic reconfiguration (FPGA) — canary rollout with 1->10->50->100% progression, auto-rollback on error threshold, identical to partial reconfiguration.
```

This module is one of 15 subsystems in the AEGIS platform. See the [unified architecture](https://github.com/KnigguKniggu-droid/AEGIS#readme) for how all components interconnect.

---

## Features

- Boolean + multivariate flags (prompt version, model, temperature, tools)
- Canary: 1% to 10% to 50% to 100% with metric gates
- Celery workers run LLM-as-judge eval on shadow traffic every 5 min
- Auto-rollback if quality metric drops >5% with p<0.01
- Slack/PagerDuty alerts; audit log for compliance

---

## Tech Stack

`Python` | `FastAPI` | `PostgreSQL` | `Redis` | `Celery` | `Slack SDK`

---

## Quick Start

```bash
git clone https://github.com/KnigguKniggu-droid/12-ai-feature-flag-system.git
cd 12-ai-feature-flag-system
pip install -e .
```

Run tests:

```bash
pytest tests/ -v
```

---

## Project Structure

```
12_ai_feature_flag_system/
  src/                  # Core application code
  tests/                # 26 passing tests
  .github/              # CI/CD workflows
  Dockerfile            # Container build
  pyproject.toml        # Package configuration
```

---

## Running in Docker

```bash
docker build -t 12_ai_feature_flag_system .
docker run -p 8000:8000 12_ai_feature_flag_system
```

---

## ECE Design Principles

This subsystem is modeled after classical electrical and computer engineering concepts:

> **Dynamic reconfiguration (FPGA) — canary rollout with 1->10->50->100% progression, auto-rollback on error threshold, identical to partial reconfiguration.**

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
