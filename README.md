# AI Engineering Workspace

A production-grade multi-project workspace containing 15 specialized AI engineering
blueprints. Each project establishes strict typed contracts, structured configuration
layers, database schemas, and multi-service deployment orchestrations.

## Projects

| # | Project | Tech Stack | Core Capability |
|---|---------|-----------|-----------------|
| 01 | Model Regression Detection System | Python, OpenAI, Pydantic, SQLite, GitHub Actions | CI/CD prompt regression testing with LLM-as-judge |
| 02 | LLM Cost Autopilot | FastAPI, OpenAI, Anthropic, Ollama, Scikit-learn | Multi-provider routing with complexity classification |
| 03 | Failure Forensics Tool | OpenTelemetry, SQLite, Streamlit | 4-step pipeline tracer with backward root cause analysis |
| 04 | Self-Healing Documentation | ChromaDB, OpenAI, PyGithub, GitHub Actions | AST parser to Markdown linker with git diff staleness |
| 05 | Output Arbitration System | LangGraph, Pydantic, Instructor, FastAPI | Parallel multi-critic judge with central adjudicator |
| 06 | Hybrid Search RAG | ChromaDB, BM25, FastAPI, Cross-Encoder | BM25 + vector fusion RAG with citation verification |
| 07 | Semantic Caching Proxy | Redis VL, FastAPI, Prometheus, Grafana | OpenAI proxy with 0.95 similarity semantic caching |
| 08 | Text-to-SQL Guardrails | SQLAlchemy, Sqlparse, FastAPI, DuckDB | SQL sandboxing with dual-query hallucination detection |
| 09 | Prompt A/B Testing | PostgreSQL, Scipy.stats, FastAPI, Streamlit | Fixed hash traffic splitting with t-test analysis |
| 10 | LoRA Fine-tuning Pipeline | PEFT, TRL, QLoRA, W&B, vLLM | LoRA training on q_proj/v_proj with forgetting checks |
| 11 | LLM API Gateway | Python, Redis, OpenTelemetry, Prometheus | Token bucket rate limiting with circuit breaker fallback |
| 12 | AI Feature Flag System | PostgreSQL, Redis, Celery, Slack | Canary rollout with LLM-as-judge and auto-rollback |
| 13 | Prod Log Dataset Generator | ClickHouse, HDBSCAN, Scikit-learn, Streamlit | Production log mining with HDBSCAN clustering |
| 14 | Agentic Workflow Orchestrator | FastAPI, SQLite | Multi-agent task dispatcher with state checkpoints |
| 15 | Realtime LLM Observability | FastAPI, WebSocket, NumPy | Streaming P95 latency and token drift dashboard |

## Architecture

Each project follows a consistent production architecture:

- **Typed Contracts**: Pydantic models defining all data structures
- **Configuration Layer**: YAML/TOML config files separating config from code
- **Database Schemas**: SQLite/PostgreSQL/ClickHouse with migration scripts
- **Service Layer**: FastAPI applications with CORS, health checks, typed endpoints
- **Deployment**: Dockerfile and docker-compose.yml for each service
- **Testing**: Unit tests for core logic with pytest
- **CI/CD**: GitHub Actions workflows where applicable

## Quick Start

```bash
# Navigate to a project
cd 01_model_regression_system

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start services (where applicable)
docker-compose up -d
```

## Environment Variables

Most projects require API keys. Create a `.env` file in each project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GITHUB_TOKEN=ghp_...
REDIS_URL=redis://localhost:6379
DATABASE_URL=duckdb:///data/analytics.duckdb
```

## Shared Infrastructure

Several projects share infrastructure services. The root docker-compose.yml
orchestrates Redis, PostgreSQL, and ClickHouse for all projects that need them.

```bash
# Start shared infrastructure
docker-compose -f docker-compose.infra.yml up -d
```

## License

MIT
