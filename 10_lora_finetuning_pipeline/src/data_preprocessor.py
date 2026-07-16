"""Dataset preprocessor with 80/10/10 split and leakage prevention.

Formats text collections into structured folds while preventing
evaluation leakage through hash-based deduplication and cross-split
content verification.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.models import DatasetItem, DatasetSplit, DatasetSplitResult

DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_instruction(item: DatasetItem, instruction_template: str = None) -> str:
    """Format a dataset item into an instruction-tuning example."""
    template = instruction_template or (
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )
    return template.format(
        instruction=item.text,
        response=item.label or item.text,
    )


def check_leakage(
    train: list[DatasetItem],
    validation: list[DatasetItem],
    test: list[DatasetItem],
) -> bool:
    """Check for evaluation leakage across splits.

    Returns True if no leakage is detected (pass), False otherwise.
    Leakage is detected when identical or near-identical items appear
    in multiple splits.
    """
    train_hashes = {hash_text(item.text) for item in train}
    val_hashes = {hash_text(item.text) for item in validation}
    test_hashes = {hash_text(item.text) for item in test}

    if train_hashes & val_hashes:
        return False
    if train_hashes & test_hashes:
        return False
    if val_hashes & test_hashes:
        return False

    train_words = {frozenset(item.text.lower().split()[:10]) for item in train if len(item.text.split()) >= 10}
    test_words = {frozenset(item.text.lower().split()[:10]) for item in test if len(item.text.split()) >= 10}
    overlap = train_words & test_words
    if len(overlap) > len(test_words) * 0.1:
        return False

    return True


def split_dataset(
    items: list[DatasetItem],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> DatasetSplitResult:
    """Split a dataset into 80/10/10 train/validation/test folds.

    Uses deterministic shuffling based on a fixed seed to ensure
    reproducible splits. Prevents evaluation leakage by checking
    for duplicate content across splits.
    """
    import random
    rng = random.Random(seed)

    seen_hashes: set[str] = set()
    unique_items: list[DatasetItem] = []
    for item in items:
        h = hash_text(item.text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_items.append(item)

    shuffled = list(unique_items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))

    train = shuffled[:train_end]
    validation = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    for item in train:
        item.split = DatasetSplit.TRAIN
    for item in validation:
        item.split = DatasetSplit.VALIDATION
    for item in test:
        item.split = DatasetSplit.TEST

    leakage_passed = check_leakage(train, validation, test)

    return DatasetSplitResult(
        train=train,
        validation=validation,
        test=test,
        split_ratios=ratios,
        total_items=n,
        leakage_check_passed=leakage_passed,
    )


def load_jsonl(file_path: str) -> list[DatasetItem]:
    """Load dataset items from a JSONL file."""
    import json
    from pathlib import Path

    items: list[DatasetItem] = []
    for line in Path(file_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            items.append(DatasetItem(
                id=data.get("id", str(len(items))),
                text=data.get("text", ""),
                label=data.get("label", ""),
                metadata=data.get("metadata", {}),
            ))
    return items
