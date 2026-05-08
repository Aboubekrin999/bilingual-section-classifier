"""
Parquet writer for unified Record streams.

Parquet is the format the training script reads — schema-checked,
fast to scan, and small on disk after dictionary encoding the
``label`` and ``language`` columns.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.datasets import Record

_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("language", pa.string()),
        ("label", pa.string()),
        ("source", pa.string()),
    ]
)


def write_records(records: Iterable[Record], path: str | Path) -> int:
    """Write records to a Parquet file. Returns the count written."""
    materialised = list(records)
    table = pa.table(
        {
            "text": [r.text for r in materialised],
            "language": [r.language for r in materialised],
            "label": [r.label.value for r in materialised],
            "source": [r.source for r in materialised],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, str(path))
    return len(materialised)


def read_records_table(path: str | Path) -> pa.Table:
    """Read a built Parquet split back as a pyarrow Table."""
    return pq.read_table(str(path))
