"""File ingestion with duplicate detection.

Parses files into document chunks, computes embeddings, and blocks
duplicates with cosine similarity > 0.95.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.models import Document, DocumentSource

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
DUPLICATE_THRESHOLD = 0.95


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def parse_file(file_path: Path) -> list[Document]:
    """Parse a single file into document chunks."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    ext = file_path.suffix.lower()
    source_type = DocumentSource.TEXT
    if ext == ".md":
        source_type = DocumentSource.MARKDOWN
    elif ext == ".py":
        source_type = DocumentSource.PYTHON
    elif ext == ".json":
        source_type = DocumentSource.JSON

    chunks = chunk_text(content)
    documents: list[Document] = []
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            doc_id=f"{file_path}:{i}",
            content=chunk,
            source_file=str(file_path),
            source_type=source_type,
            chunk_index=i,
            content_hash=hash_content(chunk),
        ))
    return documents


def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    if va.size == 0 or vb.size == 0:
        return 0.0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def detect_duplicates(documents: list[Document], threshold: float = DUPLICATE_THRESHOLD) -> list[tuple[str, str, float]]:
    """Find document pairs with cosine similarity above the threshold."""
    duplicates: list[tuple[str, str, float]] = []
    for i, doc_a in enumerate(documents):
        if doc_a.embedding is None:
            continue
        for j in range(i + 1, len(documents)):
            doc_b = documents[j]
            if doc_b.embedding is None:
                continue
            sim = cosine_sim(doc_a.embedding, doc_b.embedding)
            if sim > threshold:
                duplicates.append((doc_a.doc_id, doc_b.doc_id, sim))
    return duplicates


def deduplicate(documents: list[Document], threshold: float = DUPLICATE_THRESHOLD) -> list[Document]:
    """Remove duplicate documents, keeping the first occurrence."""
    duplicates = detect_duplicates(documents, threshold)
    duplicate_ids = {d[1] for d in duplicates}
    return [doc for doc in documents if doc.doc_id not in duplicate_ids]


def ingest_directory(root: Path, exclude: set[str] | None = None) -> list[Document]:
    """Walk a directory and parse all supported files into documents."""
    default_exclude = {".git", "__pycache__", ".venv", "node_modules"}
    exclude = exclude or default_exclude
    supported = {".md", ".py", ".json", ".txt"}
    documents: list[Document] = []
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fname in filenames:
            if Path(fname).suffix.lower() in supported:
                fpath = Path(dirpath) / fname
                documents.extend(parse_file(fpath))
    return documents
