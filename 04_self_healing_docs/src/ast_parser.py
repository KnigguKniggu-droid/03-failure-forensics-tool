"""AST repository parser that slices Python source into semantic tokens.

Uses the ast module to extract functions, classes, methods, and modules,
computing source hashes for change detection.
"""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Iterable

from src.models import CodeElementType, CodeToken


def _hash_source(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_source_segment(source: str, node: ast.AST) -> str:
    if hasattr(node, "end_lineno") and node.end_lineno and hasattr(node, "lineno"):
        lines = source.splitlines()
        start = max(0, node.lineno - 1)
        end = node.end_lineno
        return "\n".join(lines[start:end])
    return ast.get_source_segment(source, node) or ""


def _get_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = []
        for arg in node.args.args:
            annotation = ""
            if arg.annotation:
                annotation = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg.arg}{annotation}")
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        return f"def {node.name}({', '.join(args)}){returns}"
    elif isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        base_str = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{base_str}"
    return ""


def _get_docstring(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        ds = ast.get_docstring(node, clean=True)
        return ds or ""
    return ""


def parse_file(file_path: Path) -> list[CodeToken]:
    """Parse a single Python file and extract all semantic code tokens."""
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    tokens: list[CodeToken] = []
    rel_path = str(file_path)

    module_hash = _hash_source(source)
    tokens.append(CodeToken(
        element_id=f"file:{rel_path}:module:__main__",
        element_type=CodeElementType.MODULE,
        name=file_path.stem,
        file_path=rel_path,
        line_start=1,
        line_end=len(source.splitlines()),
        signature=f"module {file_path.stem}",
        docstring=_get_docstring(tree),
        source_hash=module_hash,
    ))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent_class = _find_parent_class(tree, node)
            elem_type = CodeElementType.METHOD if parent_class else CodeElementType.FUNCTION
            elem_name = f"{parent_class}.{node.name}" if parent_class else node.name
            seg = _get_source_segment(source, node)
            tokens.append(CodeToken(
                element_id=f"file:{rel_path}:{elem_type.value}:{elem_name}",
                element_type=elem_type,
                name=elem_name,
                file_path=rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                signature=_get_signature(node),
                docstring=_get_docstring(node),
                source_hash=_hash_source(seg),
                metadata={"is_async": isinstance(node, ast.AsyncFunctionDef)},
            ))
        elif isinstance(node, ast.ClassDef):
            seg = _get_source_segment(source, node)
            tokens.append(CodeToken(
                element_id=f"file:{rel_path}:class:{node.name}",
                element_type=CodeElementType.CLASS,
                name=node.name,
                file_path=rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                signature=_get_signature(node),
                docstring=_get_docstring(node),
                source_hash=_hash_source(seg),
            ))

    return tokens


def _find_parent_class(tree: ast.Module, target: ast.AST) -> str | None:
    """Find the parent class name for a method node."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if child is target:
                    return node.name
    return None


def parse_repository(root: Path, exclude: set[str] | None = None) -> list[CodeToken]:
    """Walk a repository and parse all Python files into code tokens."""
    default_exclude = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox", "build", "dist"}
    exclude = exclude or default_exclude
    tokens: list[CodeToken] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = Path(dirpath) / fname
                tokens.extend(parse_file(fpath))
    return tokens
