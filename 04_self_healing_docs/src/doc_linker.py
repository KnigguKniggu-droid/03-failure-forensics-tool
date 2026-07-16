"""Markdown parser and code-to-doc cosine similarity linker.

Parses Markdown files into semantic blocks (one per heading section)
and links code tokens to documentation blocks using cosine similarity
of their vector embeddings.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.models import CodeDocLink, CodeToken, MarkdownBlock

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_REF_PATTERN = re.compile(r"`([^`]+\.\w+)(?::([^`]+))?`")


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_markdown_file(file_path: Path) -> list[MarkdownBlock]:
    """Parse a Markdown file into semantic blocks, one per heading section."""
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    blocks: list[MarkdownBlock] = []

    current_heading = ""
    current_start = 0
    current_lines: list[str] = []
    in_code_block = False

    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        match = HEADING_PATTERN.match(line)
        if match and not in_code_block:
            if current_lines:
                block_content = "\n".join(current_lines)
                blocks.append(_make_block(file_path, current_heading, block_content, current_start, i - 1))
            current_heading = match.group(2).strip()
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        block_content = "\n".join(current_lines)
        blocks.append(_make_block(file_path, current_heading, block_content, current_start, len(lines) - 1))

    return blocks


def _make_block(file_path: Path, heading: str, content: str, start: int, end: int) -> MarkdownBlock:
    code_refs = CODE_REF_PATTERN.findall(content)
    linked_path = code_refs[0][0] if code_refs else None
    linked_elem = code_refs[0][1] if code_refs and code_refs[0][1] else None
    return MarkdownBlock(
        block_id=f"md:{file_path}:{heading}:{_hash_content(content)[:8]}",
        file_path=str(file_path),
        heading=heading,
        content=content,
        line_start=start,
        line_end=end,
        linked_code_path=linked_path,
        linked_element=linked_elem,
        source_hash=_hash_content(content),
    )


def parse_markdown_repository(root: Path, exclude: set[str] | None = None) -> list[MarkdownBlock]:
    default_exclude = {".git", "__pycache__", ".venv", "node_modules"}
    exclude = exclude or default_exclude
    blocks: list[MarkdownBlock] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fname in filenames:
            if fname.endswith(".md"):
                fpath = Path(dirpath) / fname
                blocks.extend(parse_markdown_file(fpath))
    return blocks


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def link_code_to_docs(
    code_tokens: list[CodeToken],
    markdown_blocks: list[MarkdownBlock],
    similarity_threshold: float = 0.5,
) -> list[CodeDocLink]:
    """Link code tokens to markdown blocks using cosine similarity.

    Only links where cosine similarity exceeds the threshold are returned.
    Links with similarity > 0.95 are flagged as potential duplicates.
    """
    links: list[CodeDocLink] = []
    for token in code_tokens:
        if token.embedding is None:
            continue
        for block in markdown_blocks:
            if block.embedding is None:
                continue
            sim = cosine_similarity(token.embedding, block.embedding)
            if sim >= similarity_threshold:
                links.append(CodeDocLink(
                    code_token_id=token.element_id,
                    markdown_block_id=block.block_id,
                    cosine_similarity=sim,
                ))
    links.sort(key=lambda l: l.cosine_similarity, reverse=True)
    return links


def detect_duplicates(links: list[CodeDocLink], threshold: float = 0.95) -> list[CodeDocLink]:
    """Flag links with cosine similarity above the duplicate threshold."""
    return [l for l in links if l.cosine_similarity > threshold]
