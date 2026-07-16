"""Tests for the AST parser and documentation linker."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.ast_parser import parse_file, parse_repository
from src.doc_linker import parse_markdown_file, cosine_similarity, parse_markdown_repository
from src.models import CodeElementType
from src.diff_checker import parse_git_diff


SAMPLE_PY = '''"""Sample module for testing."""

def hello(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}"


class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b
'''


SAMPLE_MD = '''# Calculator Module

This module provides arithmetic operations.

## Functions

The `hello` function greets a person.

## Calculator Class

The `Calculator` class in `src/calculator.py` provides:
- `add` method
- `subtract` method
'''


def test_parse_file_extracts_functions_and_classes():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_PY)
        f.flush()
        path = Path(f.name)

    tokens = parse_file(path)
    types = {t.element_type for t in tokens}
    assert CodeElementType.MODULE in types
    assert CodeElementType.FUNCTION in types
    assert CodeElementType.CLASS in types
    assert CodeElementType.METHOD in types

    names = [t.name for t in tokens]
    assert "hello" in names
    assert "Calculator" in names
    assert "Calculator.add" in names
    assert "Calculator.subtract" in names
    path.unlink()


def test_parse_file_computes_source_hashes():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_PY)
        f.flush()
        path = Path(f.name)

    tokens = parse_file(path)
    for t in tokens:
        assert len(t.source_hash) == 64
        assert t.line_start > 0
        assert t.line_end >= t.line_start
    path.unlink()


def test_parse_repository_walks_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def foo(): pass\n", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "b.py").write_text("class Bar: pass\n", encoding="utf-8")
        tokens = parse_repository(root)
        names = [t.name for t in tokens]
        assert "foo" in names
        assert "Bar" in names


def test_parse_markdown_file_extracts_blocks():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_MD)
        f.flush()
        path = Path(f.name)

    blocks = parse_markdown_file(path)
    assert len(blocks) >= 3
    headings = [b.heading for b in blocks]
    assert "Calculator Module" in headings
    assert "Functions" in headings
    assert "Calculator Class" in headings
    path.unlink()


def test_cosine_similarity_identical_vectors():
    vec = [1.0, 2.0, 3.0]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_empty_vectors():
    assert cosine_similarity([], [1.0]) == 0.0


def test_parse_git_diff_extracts_file_entries():
    diff = """diff --git a/src/foo.py b/src/foo.py
index 1234567..abcdefg 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,3 +10,4 @@
 def foo():
-    return 1
+    return 2
+    # new line
"""
    entries = parse_git_diff(diff)
    assert len(entries) == 1
    assert entries[0].file_path == "src/foo.py"
    assert entries[0].change_type == "modified"
    assert len(entries[0].removed_lines) == 1
