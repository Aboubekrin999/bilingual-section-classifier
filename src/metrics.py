"""
Pure classification metrics: per-class F1, per-language F1, confusion matrix.

Implemented from scratch (no scikit-learn dep at import time) so this
module loads in CI without the heavy ML stack. The training script
will use scikit-learn directly for speed; this module is the source
of truth for the *evaluation report*, not for the training loop.

The headline number for the bilingual classifier is **macro-F1 averaged
within each language** — pooled metrics hide language-specific failure
modes the project exists to fix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PerClassF1:
    """Precision / recall / F1 / support for one class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationReport:
    """Aggregated metrics over one (sub)set of predictions."""

    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class: list[PerClassF1]


def precision_recall_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label: str,
) -> tuple[float, float, float, int]:
    """Per-class precision, recall, F1, and support against one label."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must align"
        )
    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == label and yp == label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt != label and yp == label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == label and yp != label)
    support = sum(1 for yt in y_true if yt == label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1, support


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> ClassificationReport:
    """Per-class F1 + macro / weighted F1 + accuracy."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must align"
        )
    if not labels:
        raise ValueError("at least one label is required")

    per_class: list[PerClassF1] = []
    for label in labels:
        p, r, f, s = precision_recall_f1(y_true, y_pred, label)
        per_class.append(PerClassF1(label=label, precision=p, recall=r, f1=f, support=s))

    macro_f1 = sum(c.f1 for c in per_class) / len(per_class)
    total_support = sum(c.support for c in per_class)
    weighted_f1 = (
        sum(c.f1 * c.support for c in per_class) / total_support
        if total_support
        else 0.0
    )
    accuracy = (
        sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == yp) / len(y_true)
        if y_true
        else 0.0
    )
    return ClassificationReport(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=accuracy,
        per_class=per_class,
    )


def per_language_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    languages: Sequence[str],
    labels: Sequence[str],
) -> dict[str, ClassificationReport]:
    """Slice predictions by language and report each separately.

    The headline metric for this project is the macro-F1 of the worst
    language, not pooled accuracy — surfacing that requires this slice.
    """
    if not (len(y_true) == len(y_pred) == len(languages)):
        raise ValueError(
            f"y_true ({len(y_true)}), y_pred ({len(y_pred)}), and "
            f"languages ({len(languages)}) must all align"
        )
    by_lang: dict[str, list[tuple[str, str]]] = {}
    for yt, yp, lang in zip(y_true, y_pred, languages, strict=False):
        by_lang.setdefault(lang, []).append((yt, yp))

    reports: dict[str, ClassificationReport] = {}
    for lang, pairs in by_lang.items():
        sub_true = [t for t, _ in pairs]
        sub_pred = [p for _, p in pairs]
        reports[lang] = classification_report(sub_true, sub_pred, labels)
    return reports


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> list[list[int]]:
    """N x N counts where entry [i][j] is "true label i predicted as label j"."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must align"
        )
    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred, strict=False):
        if yt in index and yp in index:
            matrix[index[yt]][index[yp]] += 1
    return matrix
