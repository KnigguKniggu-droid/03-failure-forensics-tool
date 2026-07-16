"""LLM reconciliation script for documentation updates.

Compares old code against new code changes and produces structural
file edit diff patches for outdated documentation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from src.models import CodeToken, DocPatch, MarkdownBlock, ReconciliationResult, StalenessReport

RECONCILIATION_PROMPT = """You are a technical documentation reconciler.
Given the old code, new code, and the current documentation, determine if
the documentation is stale and produce an updated version.

Old code:
{old_code}

New code:
{new_code}

Current documentation:
{current_doc}

Respond in JSON with this schema:
{{
  "needs_update": true/false,
  "new_content": "the updated markdown content",
  "reasoning": "why this update is needed",
  "confidence": 0.0-1.0
}}"""


async def reconcile_block(
    old_code: str,
    new_code: str,
    current_doc: str,
    block_heading: str,
    file_path: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> DocPatch | None:
    """Use an LLM to reconcile a single documentation block against code changes."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    prompt = RECONCILIATION_PROMPT.format(
        old_code=old_code[:3000],
        new_code=new_code[:3000],
        current_doc=current_doc[:3000],
    )

    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a precise documentation reconciler."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)

        if not parsed.get("needs_update", False):
            return None

        return DocPatch(
            file_path=file_path,
            patch_type="update",
            block_heading=block_heading,
            old_content=current_doc,
            new_content=parsed.get("new_content", ""),
            reasoning=parsed.get("reasoning", ""),
            confidence=float(parsed.get("confidence", 0.0)),
        )
    except Exception:
        return None


async def reconcile_stale_docs(
    staleness_report: StalenessReport,
    code_tokens: list[CodeToken],
    markdown_blocks: list[MarkdownBlock],
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> ReconciliationResult:
    """Run LLM reconciliation across all stale documentation blocks."""
    patches: list[DocPatch] = []

    token_by_id = {t.element_id: t for t in code_tokens}
    block_by_id = {b.block_id: b for b in markdown_blocks}

    for block in staleness_report.affected_markdown_blocks:
        linked_token = None
        for token in staleness_report.affected_code_tokens:
            if block.linked_element and block.linked_element in token.name:
                linked_token = token
                break

        if linked_token is None:
            continue

        old_code = linked_token.docstring or linked_token.signature
        new_code = linked_token.signature
        patch = await reconcile_block(
            old_code=old_code,
            new_code=new_code,
            current_doc=block.content,
            block_heading=block.heading,
            file_path=block.file_path,
            model=model,
            api_key=api_key,
        )
        if patch:
            patches.append(patch)

    overall = len(patches) / max(1, len(staleness_report.affected_markdown_blocks))

    return ReconciliationResult(
        patches=patches,
        overall_staleness=overall,
        summary=f"Reconciled {len(patches)} documentation block(s) across {len(staleness_report.affected_markdown_blocks)} affected block(s)",
        model_used=model,
    )


def format_patch_as_unified(patch: DocPatch) -> str:
    """Format a DocPatch as a unified diff string."""
    old_lines = patch.old_content.splitlines()
    new_lines = patch.new_content.splitlines()
    diff_lines = [f"--- a/{patch.file_path}", f"+++ b/{patch.file_path}"]
    diff_lines.append(f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@")
    for line in old_lines:
        diff_lines.append(f"-{line}")
    for line in new_lines:
        diff_lines.append(f"+{line}")
    return "\n".join(diff_lines)
