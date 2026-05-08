"""
Fine-tune XLM-RoBERTa-base on the built bilingual section corpus.

Designed to run on Colab (T4 / A100). The CI environment intentionally
does not have torch/transformers installed; this file lints under ruff
but isn't imported at test time.

Usage on Colab::

    !pip install -q torch transformers datasets accelerate \\
                    scikit-learn pyarrow wandb sentencepiece
    %cd /content/bilingual-section-classifier
    !python -m scripts.train \\
        --output-dir runs/baseline \\
        --wandb-project bilingual-section \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset, DatasetDict
from src.labels import all_canonical_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("--data-dir", type=Path, default=Path("data/built"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _load_split(path: Path) -> Dataset:
    table = pq.read_table(str(path))
    return Dataset(table)


def main() -> None:
    args = parse_args()

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    splits = DatasetDict(
        {
            "train": _load_split(args.data_dir / "train.parquet"),
            "val": _load_split(args.data_dir / "val.parquet"),
            "test": _load_split(args.data_dir / "test.parquet"),
        }
    )

    label_names = all_canonical_labels()
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch: dict) -> dict:
        encoding = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )
        encoding["labels"] = [label2id[label] for label in batch["label"]]
        return encoding

    tokenised = splits.map(
        tokenize,
        batched=True,
        remove_columns=["text", "label", "language", "source"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    def compute_metrics(eval_pred) -> dict[str, float]:
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        return {
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["val"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=tokenised["test"], metric_key_prefix="test")
    print("\n=== Test set ===")
    for key in sorted(test_metrics):
        print(f"  {key:30s} {test_metrics[key]:.4f}")

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved best checkpoint to {final_dir}/")


if __name__ == "__main__":
    main()
