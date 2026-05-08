"""
Score a trained checkpoint against the test set.

Loads the model, runs predictions, and prints:

- Overall macro / weighted F1 + accuracy
- Per-class precision / recall / F1 / support
- **Per-language F1** (the headline metric for this project)
- Confusion matrix

Usage on Colab::

    !python -m scripts.evaluate \\
        --model-dir runs/baseline/final \\
        --data-dir data/built
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.labels import all_canonical_labels
from src.metrics import (
    classification_report,
    confusion_matrix,
    per_language_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/built"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    table = pq.read_table(str(args.data_dir / f"{args.split}.parquet"))
    texts = table.column("text").to_pylist()
    y_true = table.column("label").to_pylist()
    languages = table.column("language").to_pylist()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    model.to(args.device)
    model.eval()

    id2label = model.config.id2label

    y_pred: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            logits = model(**encoded).logits
            preds = logits.argmax(dim=-1).tolist()
            y_pred.extend(id2label[p] for p in preds)

    label_names = all_canonical_labels()
    overall = classification_report(y_true, y_pred, label_names)
    by_lang = per_language_report(y_true, y_pred, languages, label_names)
    matrix = confusion_matrix(y_true, y_pred, label_names)

    print(f"\n=== {args.split} split — overall ===")
    print(f"  accuracy:    {overall.accuracy:.4f}")
    print(f"  macro_f1:    {overall.macro_f1:.4f}")
    print(f"  weighted_f1: {overall.weighted_f1:.4f}")

    print("\n  per-class:")
    print(f"    {'label':16s} {'P':>8s} {'R':>8s} {'F1':>8s} {'support':>8s}")
    for c in overall.per_class:
        print(
            f"    {c.label:16s} {c.precision:8.4f} {c.recall:8.4f} "
            f"{c.f1:8.4f} {c.support:8d}"
        )

    print("\n=== Per-language ===")
    for lang in sorted(by_lang):
        report = by_lang[lang]
        print(
            f"  {lang}: macro_f1={report.macro_f1:.4f} "
            f"acc={report.accuracy:.4f} n={sum(c.support for c in report.per_class)}"
        )

    print("\n=== Confusion matrix (rows = true, cols = predicted) ===")
    header = "          " + "".join(f"{label[:6]:>8s}" for label in label_names)
    print(header)
    for label, row in zip(label_names, matrix, strict=False):
        cells = "".join(f"{count:>8d}" for count in row)
        print(f"  {label:8s}{cells}")


if __name__ == "__main__":
    main()
