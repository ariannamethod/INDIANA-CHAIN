"""Utilities for loading datasets from local files or HuggingFace hub."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from datasets import load_dataset as hf_load_dataset


def load_dataset(source: str, split: str = "train") -> Iterable[Dict[str, Any]]:
    """Yield samples from ``source``.

    ``source`` may either be a path to a local JSONL/JSON/CSV file or the name of a
    dataset on the HuggingFace hub. When a HuggingFace dataset name is provided the
    ``split`` argument selects which split to stream.
    """

    path = Path(source)
    if path.exists():
        if path.suffix in {".jsonl", ".json"}:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
        else:
            raise ValueError(f"Unsupported dataset extension: {path.suffix}")
    else:
        ds = hf_load_dataset(source, split=split)
        for item in ds:
            yield item
