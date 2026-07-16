"""HDBSCAN clustering engine for production log semantic grouping.

Clusters production logs by semantic similarity using HDBSCAN with
scikit-learn preprocessing. Identifies coherent groups for dataset sampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models import Cluster, ProductionLog

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from sklearn.decomposition import PCA

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_embeddings(logs: list[ProductionLog], use_tfidf: bool = True) -> np.ndarray:
    """Compute or extract embeddings for all logs.

    If logs have pre-computed embeddings, use those. Otherwise, fall back
    to TF-IDF vectorization for semantic representation.
    """
    embedded = [log for log in logs if log.embedding is not None]
    if embedded and len(embedded) == len(logs):
        return np.array([log.embedding for log in logs], dtype=np.float64)

    if not HAS_SKLEARN:
        return np.array([[hash(log.user_query) % 1000 / 1000.0] for log in logs], dtype=np.float64)

    queries = [log.user_query for log in logs]
    vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(queries)
    return tfidf_matrix.toarray().astype(np.float64)


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Reduce embedding dimensions using PCA for HDBSCAN efficiency."""
    if embeddings.shape[1] <= n_components:
        return embeddings
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def cluster_logs(
    logs: list[ProductionLog],
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> list[Cluster]:
    """Cluster production logs using HDBSCAN.

    Returns a list of Cluster objects, including a noise cluster (label -1)
    for logs that do not fit any cluster.
    """
    if len(logs) < min_cluster_size:
        return [Cluster(
            cluster_id=-1,
            label="noise",
            log_ids=[log.log_id for log in logs],
            size=len(logs),
        )]

    embeddings = compute_embeddings(logs)
    reduced = reduce_dimensions(embeddings, n_components=min(10, embeddings.shape[1]))

    if HAS_HDBSCAN:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced)
    else:
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=min_samples)
        labels = clusterer.fit_predict(reduced)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    result: list[Cluster] = []
    for label, indices in clusters.items():
        cluster_logs_subset = [logs[i] for i in indices]
        centroid = np.mean(reduced[indices], axis=0) if indices else None

        representative_idx = indices[0]
        if centroid is not None:
            distances = np.linalg.norm(reduced[indices] - centroid, axis=1)
            representative_idx = indices[int(np.argmin(distances))]

        keywords = _extract_keywords([logs[i].user_query for i in indices])
        coherence = _compute_coherence(reduced[indices])

        result.append(Cluster(
            cluster_id=label,
            label="noise" if label == -1 else f"cluster_{label}",
            log_ids=[logs[i].log_id for i in indices],
            centroid_embedding=centroid.tolist() if centroid is not None else None,
            size=len(indices),
            coherence_score=coherence,
            representative_query=logs[representative_idx].user_query,
            keywords=keywords,
        ))

    result.sort(key=lambda c: c.size, reverse=True)
    return result


def _extract_keywords(queries: list[str], top_n: int = 5) -> list[str]:
    """Extract top keywords from a set of queries."""
    if not HAS_SKLEARN or not queries:
        return []
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english")
    try:
        matrix = vectorizer.fit_transform(queries)
        scores = np.array(matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [vectorizer.get_feature_names_out()[i] for i in top_indices if scores[i] > 0]
    except Exception:
        return []


def _compute_coherence(embeddings: np.ndarray) -> float:
    """Compute intra-cluster coherence as 1 - normalized average distance."""
    if len(embeddings) < 2:
        return 1.0
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    avg_dist = float(np.mean(distances))
    max_possible = float(np.linalg.norm(np.max(embeddings, axis=0) - np.min(embeddings, axis=0)))
    if max_possible == 0:
        return 1.0
    return max(0.0, 1.0 - avg_dist / max_possible)
