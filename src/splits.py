"""
Stratified train/val/test splitter.

Stratifies by (label × language) so every split preserves both the class
distribution and the EN/FR balance — critical for a bilingual classifier
where macro-F1 per language is the headline metric.

Implemented from scratch (no scikit-learn at import time) to keep CI dep
footprint zero. The tradeoff is that we don't get sklearn's edge-case
handling for tiny groups; instead we follow a simple rule: when a group
has fewer than three records, every record goes to ``train`` so neither
val nor test ever has zero coverage of a (label × language) cell.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from src.datasets import Record


@dataclass(frozen=True)
class SplitResult:
    """Three disjoint Record lists summing to the input length."""

    train: list[Record]
    val: list[Record]
    test: list[Record]


def stratified_split(
    records: Sequence[Record],
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 13,
) -> SplitResult:
    """
    Split ``records`` into train/val/test, stratified by (label, language).

    The split is deterministic for a given seed. ``val_ratio + test_ratio``
    must be < 1; the remainder goes to train. Stratification groups with
    fewer than 3 records are placed entirely into train so val and test
    never miss a (label × language) cell that train has.
    """
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be < 1"
        )

    rng = random.Random(seed)

    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.label.value, r.language)].append(r)

    train: list[Record] = []
    val: list[Record] = []
    test: list[Record] = []

    # Iterate groups in a deterministic order so the seed fully controls output.
    for key in sorted(groups):
        group = groups[key]
        # Shuffle within the group so val/test pull from a representative
        # spread rather than the dataset's natural ordering (which is often
        # by document and would leak document-level structure into val).
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n < 3:
            train.extend(shuffled)
            continue

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        # Guard against pathological ratios eating all of train.
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1

        test.extend(shuffled[:n_test])
        val.extend(shuffled[n_test : n_test + n_val])
        train.extend(shuffled[n_test + n_val :])

    return SplitResult(train=train, val=val, test=test)
