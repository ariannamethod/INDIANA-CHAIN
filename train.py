"""Simple fine-tuning utility for the Indiana-C model."""

from __future__ import annotations

import argparse
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from data_utils import load_dataset
from indiana_core import IndianaC, IndianaCConfig, tokenizer


class TextDataset(Dataset):
    """Tokenized dataset for supervised fine-tuning."""

    def __init__(self, data: List[dict], prompt_key: str, response_key: str):
        self.samples: List[torch.Tensor] = []
        for item in data:
            text = f"{item[prompt_key]}\n{item[response_key]}"
            tokens = tokenizer.encode(text).squeeze(0)
            self.samples.append(tokens)

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.samples[idx]
        return tokens[:-1], tokens[1:]


def collate(batch: List[tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_batch = torch.stack(
        [torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.long)]) for x in xs]
    )
    y_batch = torch.stack(
        [torch.cat([y, torch.zeros(max_len - y.size(0), dtype=torch.long)]) for y in ys]
    )
    return x_batch, y_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Indiana-C on a dataset")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--response-key", default="answer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save", default="finetuned.pt", help="Model checkpoint path")
    args = parser.parse_args()

    data_list = list(load_dataset(args.dataset, split=args.split))
    dataset = TextDataset(data_list, args.prompt_key, args.response_key)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = IndianaC(IndianaCConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for _ in range(args.epochs):
        for x, y in loader:
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), args.save)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
