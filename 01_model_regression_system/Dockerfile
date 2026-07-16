FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

RUN pytest tests/ -v --tb=short

CMD ["python", "-m", "src.cli", "--prompt", "prompts/classifier_v1.yaml", "--tests", "tests/ground_truth.json"]
