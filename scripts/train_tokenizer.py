from __future__ import annotations

import csv
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def iter_texts() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    datasets_dir = root / "datasets"
    for path in datasets_dir.glob("*"):
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    for value in item.values():
                        if isinstance(value, str):
                            yield value
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        if key.lower() in {"line", "text"} and value:
                            yield value
        else:
            continue


def main() -> None:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(iter_texts(), trainer)
    out_path = Path(__file__).resolve().parents[1] / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer to {out_path}")


if __name__ == "__main__":
    main()
