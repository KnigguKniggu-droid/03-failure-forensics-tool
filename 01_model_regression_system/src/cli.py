"""CLI entry point for the regression detection pipeline.

Usage:
    python -m src.cli --prompt prompts/classifier_v1.yaml --tests tests/ground_truth.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from src.db import get_connection, init_schema, save_prompt_config, get_baseline, save_run, set_baseline
from src.evaluator import judge_batch
from src.models import PromptConfig
from src.regressor import (
    LLMClient,
    build_report,
    load_ground_truth,
    run_predictions_batched,
)

console = Console()


def load_prompt_config(path: Path) -> PromptConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptConfig.model_validate(raw)


async def run_regression(
    config: PromptConfig,
    test_path: Path,
    api_key: str | None,
    set_new_baseline: bool = False,
) -> None:
    conn = get_connection()
    init_schema(conn)
    save_prompt_config(conn, config)

    items = load_ground_truth(test_path)
    console.print(f"[bold]Loaded {len(items)} ground-truth items[/bold]")

    llm = LLMClient(api_key=api_key)
    predictions = await run_predictions_batched(config, items, llm)

    scoring = await judge_batch(items, predictions, api_key)

    baseline = get_baseline(conn, config.prompt_id, config.version)
    if baseline and not set_new_baseline:
        baseline_acc = baseline["accuracy"]
        baseline_judge = baseline["mean_judge_relevance"]
    else:
        baseline_acc = 0.0
        baseline_judge = 0.0

    report = build_report(config, items, predictions, scoring, baseline_acc, baseline_judge)
    save_run(conn, report)

    if set_new_baseline:
        set_baseline(conn, report)
        console.print("[green]New baseline recorded.[/green]")

    _print_report(report)

    if report.blocks_merge:
        console.print("[bold red]REGRESSION CRITICAL: merge blocked[/bold red]")
        sys.exit(1)
    elif report.severity.value == "warning":
        console.print("[bold yellow]REGRESSION WARNING: review before merge[/bold yellow]")

    conn.close()


def _print_report(report) -> None:
    table = Table(title=f"Regression Report: {report.prompt_id} v{report.prompt_version}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Model", report.model)
    table.add_row("Accuracy", f"{report.accuracy:.4f}")
    table.add_row("Baseline", f"{report.baseline_accuracy:.4f}")
    table.add_row("Delta", f"{report.regression_delta:+.4f}")
    table.add_row("Severity", report.severity.value.upper())
    table.add_row("Mean Judge Relevance", f"{report.mean_judge_relevance:.4f}")
    table.add_row("Total Items", str(report.total_items))
    table.add_row("Correct", str(report.correct_count))
    table.add_row("Blocks Merge", str(report.blocks_merge))
    console.print(table)

    if report.failed_items:
        fail_table = Table(title="Failed Items")
        fail_table.add_column("ID")
        fail_table.add_column("Expected")
        fail_table.add_column("Predicted")
        fail_table.add_column("Judge Score")
        fail_table.add_column("Difficulty")
        for f in report.failed_items:
            fail_table.add_row(
                f.item_id,
                f.expected_category.value,
                f.predicted_category.value if f.predicted_category else "NONE",
                f"{f.judge_relevance_score:.2f}",
                f.difficulty.value,
            )
        console.print(fail_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt regression detection")
    parser.add_argument("--prompt", required=True, help="Path to prompt YAML file")
    parser.add_argument("--tests", required=True, help="Path to ground-truth JSON file")
    parser.add_argument("--set-baseline", action="store_true", help="Record this run as the new baseline")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    args = parser.parse_args()

    config = load_prompt_config(Path(args.prompt))
    asyncio.run(
        run_regression(
            config,
            Path(args.tests),
            args.api_key,
            set_new_baseline=args.set_baseline,
        )
    )


if __name__ == "__main__":
    main()
