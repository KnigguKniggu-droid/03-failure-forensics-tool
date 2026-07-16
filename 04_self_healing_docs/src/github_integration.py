"""GitHub PR integration using PyGithub.

Fetches active PR diffs and posts documentation update comments.
"""

from __future__ import annotations

import os
from typing import Any

from src.models import ReconciliationResult

try:
    from github import Github
    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False


def get_github_client(token: str | None = None) -> Any:
    """Create a PyGithub client from a personal access token."""
    if not HAS_GITHUB:
        raise RuntimeError("PyGithub is not installed. Run: pip install PyGithub")
    token = token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is not set")
    return Github(token)


def get_pr_diff_files(repo_name: str, pr_number: int, token: str | None = None) -> list[dict[str, Any]]:
    """Fetch the list of changed files in a PR."""
    client = get_github_client(token)
    repo = client.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    files = pr.get_files()
    return [
        {
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": f.patch or "",
        }
        for f in files
    ]


def post_doc_review_comment(
    repo_name: str,
    pr_number: int,
    result: ReconciliationResult,
    token: str | None = None,
) -> None:
    """Post a review comment summarizing documentation staleness findings."""
    client = get_github_client(token)
    repo = client.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    lines = [
        f"## Self-Healing Documentation Report",
        f"",
        f"**Staleness Score**: {result.overall_staleness:.1%}",
        f"**Patches Generated**: {len(result.patches)}",
        f"**Model**: {result.model_used}",
        f"",
        f"{result.summary}",
        f"",
    ]

    for patch in result.patches:
        lines.append(f"### {patch.patch_type.title()}: `{patch.block_heading}` in `{patch.file_path}`")
        lines.append(f"**Confidence**: {patch.confidence:.0%}")
        lines.append(f"**Reasoning**: {patch.reasoning}")
        lines.append(f"")
        lines.append(f"```diff")
        lines.append(f"--- a/{patch.file_path}")
        lines.append(f"+++ b/{patch.file_path}")
        old_lines = patch.old_content.splitlines()[:10]
        new_lines = patch.new_content.splitlines()[:10]
        for ol in old_lines:
            lines.append(f"-{ol}")
        for nl in new_lines:
            lines.append(f"+{nl}")
        lines.append(f"```")
        lines.append(f"")

    pr.create_issue_comment("\n".join(lines))
