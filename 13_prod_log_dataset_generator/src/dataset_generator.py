"""Evaluation dataset generator from clustered production logs.

Samples diverse items from each cluster to generate a balanced evaluation
dataset with coverage across all identified production patterns.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from src.models import (
    Cluster,
    EvalDataset,
    EvalDatasetItem,
    LogSource,
    MiningConfig,
    ProductionLog,
)
from src.clustering import cluster_logs


def classify_difficulty(cluster: Cluster, logs: list[ProductionLog]) -> str:
    """Classify the difficulty of a cluster based on log characteristics."""
    cluster_logs = [log for log in logs if log.log_id in cluster.log_ids]
    if not cluster_logs:
        return "medium"

    avg_tokens = sum(log.input_tokens for log in cluster_logs) / len(cluster_logs)
    avg_latency = sum(log.latency_ms for log in cluster_logs) / len(cluster_logs)
    error_rate = sum(1 for log in cluster_logs if not log.success) / len(cluster_logs)

    score = 0.0
    if avg_tokens > 300:
        score += 1
    if avg_latency > 2000:
        score += 1
    if error_rate > 0.1:
        score += 1
    if cluster.coherence_score < 0.5:
        score += 1

    if score >= 3:
        return "hard"
    elif score >= 1:
        return "medium"
    return "easy"


def sample_from_cluster(
    cluster: Cluster,
    logs: list[ProductionLog],
    items_per_cluster: int = 3,
) -> list[EvalDatasetItem]:
    """Sample diverse items from a cluster for the evaluation dataset."""
    if cluster.cluster_id == -1 or cluster.size < 2:
        return []

    cluster_logs = [log for log in logs if log.log_id in cluster.log_ids]
    if not cluster_logs:
        return []

    n = min(items_per_cluster, len(cluster_logs))
    sampled = random.sample(cluster_logs, n) if len(cluster_logs) > n else cluster_logs
    difficulty = classify_difficulty(cluster, logs)

    items: list[EvalDatasetItem] = []
    for log in sampled:
        items.append(EvalDatasetItem(
            item_id=str(uuid.uuid4()),
            query=log.user_query,
            expected_response=log.model_response,
            cluster_id=cluster.cluster_id,
            difficulty=difficulty,
            source_log_id=log.log_id,
            metadata={
                "model": log.model,
                "input_tokens": log.input_tokens,
                "latency_ms": log.latency_ms,
                "keywords": cluster.keywords,
            },
        ))
    return items


def generate_dataset(
    logs: list[ProductionLog],
    config: MiningConfig,
) -> EvalDataset:
    """Generate a complete evaluation dataset from production logs."""
    random.seed(42)

    clusters = cluster_logs(
        logs,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
    )

    all_items: list[EvalDatasetItem] = []
    non_noise_clusters = [c for c in clusters if c.cluster_id != -1]
    noise_points = sum(c.size for c in clusters if c.cluster_id == -1)

    for cluster in non_noise_clusters:
        items = sample_from_cluster(cluster, logs, config.items_per_cluster)
        all_items.extend(items)

    coverage = len(all_items) / max(1, len(logs) - noise_points)

    return EvalDataset(
        dataset_id=str(uuid.uuid4()),
        name=f"eval_dataset_{len(all_items)}_items",
        description=f"Generated from {len(logs)} production logs across {len(non_noise_clusters)} clusters",
        items=all_items,
        source=config.source,
        total_logs_mined=len(logs),
        total_clusters=len(non_noise_clusters),
        noise_points=noise_points,
        coverage_score=min(1.0, coverage),
        clustering_params={
            "min_cluster_size": config.min_cluster_size,
            "min_samples": config.min_samples,
            "items_per_cluster": config.items_per_cluster,
        },
    )


def export_dataset(dataset: EvalDataset, output_path: str) -> None:
    """Export the evaluation dataset to a JSON file."""
    from pathlib import Path
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
