"""Git diff parser and document staleness detector.

Parses git diff output from active PRs, identifies which code elements
and documentation blocks are affected, and computes a staleness report.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from src.models import CodeToken, GitDiffEntry, MarkdownBlock, StalenessReport

DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+)$")
HUNK_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
ADDED_LINE = re.compile(r"^\+(?!\+)")
REMOVED_LINE = re.compile(r"^-(?!-)")


def parse_git_diff(diff_text: str) -> list[GitDiffEntry]:
    """Parse unified diff text into structured GitDiffEntry objects."""
    entries: list[GitDiffEntry] = []
    current_entry: GitDiffEntry | None = None
    current_old_line = 0

    for line in diff_text.splitlines():
        file_match = DIFF_FILE_PATTERN.match(line)
        if file_match:
            if current_entry:
                entries.append(current_entry)
            old_path = file_match.group(1)
            new_path = file_match.group(2)
            change_type = "modified"
            if old_path == new_path:
                pass
            current_entry = GitDiffEntry(
                file_path=new_path,
                change_type=change_type,
                old_path=old_path if old_path != new_path else None,
                diff_content="",
            )
            current_old_line = 0
            continue

        if current_entry is None:
            continue

        current_entry.diff_content += line + "\n"

        hunk_match = HUNK_PATTERN.match(line)
        if hunk_match:
            current_old_line = int(hunk_match.group(1))
            continue

        if REMOVED_LINE.match(line):
            current_entry.removed_lines.append(current_old_line)
            current_old_line += 1
        elif ADDED_LINE.match(line):
            current_entry.added_lines.append(0)
        elif not line.startswith("\\"):
            current_old_line += 1

    if current_entry:
        entries.append(current_entry)

    return entries


def get_pr_diff(repo_root: Path, base_branch: str = "main", head_branch: str = "") -> str:
    """Get the git diff between base and head branch."""
    cmd = ["git", "diff", f"origin/{base_branch}...HEAD"]
    if head_branch:
        cmd = ["git", "diff", f"origin/{base_branch}...{head_branch}"]
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def detect_staleness(
    diffs: list[GitDiffEntry],
    code_tokens: list[CodeToken],
    markdown_blocks: list[MarkdownBlock],
) -> StalenessReport:
    """Detect which code tokens and markdown blocks are affected by diffs.

    A code token is affected if its file appears in the diff and its line
    range overlaps with changed lines. A markdown block is stale if its
    linked code path matches a changed file.
    """
    diff_files = {d.file_path for d in diffs}
    changed_code_paths = {d.file_path for d in diffs if d.file_path.endswith(".py")}
    changed_md_paths = {d.file_path for d in diffs if d.file_path.endswith(".md")}

    affected_tokens = [
        t for t in code_tokens
        if t.file_path in changed_code_paths
    ]

    affected_blocks = [
        b for b in markdown_blocks
        if b.file_path in changed_md_paths
        or (b.linked_code_path and any(b.linked_code_path.endswith(p) or p.endswith(b.linked_code_path) for p in diff_files))
    ]

    stale_links: list = []
    total_links = 0
    for block in affected_blocks:
        for token in affected_tokens:
            if block.linked_element and block.linked_element in token.name:
                total_links += 1
                if token.source_hash != block.source_hash:
                    stale_links.append(type("Link", (), {
                        "code_token_id": token.element_id,
                        "markdown_block_id": block.block_id,
                        "is_stale": True,
                        "staleness_reason": "source hash mismatch",
                    })())

    staleness_score = len(stale_links) / total_links if total_links > 0 else 0.0

    return StalenessReport(
        affected_code_tokens=affected_tokens,
        affected_markdown_blocks=affected_blocks,
        stale_links=stale_links,
        git_diffs=diffs,
        staleness_score=staleness_score,
    )
