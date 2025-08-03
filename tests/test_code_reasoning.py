"""Tests for code generation and evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from indiana_core import generate_consistent_text

from code_eval import code_eval


def test_code_tasks_execute_correctly() -> None:
    dataset_path = (
        Path(__file__).resolve().parent.parent / "datasets" / "code_subset.jsonl"
    )
    samples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]

    # Map each prompt to a snippet that simply sets ``result`` to the expected value.
    solutions = {sample["prompt"]: f"result = {repr(sample['expected'])}" for sample in samples}

    def fake_generate(prompt: str, **kwargs) -> str:
        return solutions[prompt]

    with patch("tests.test_code_reasoning.generate_consistent_text", side_effect=fake_generate):
        for sample in samples:
            code = generate_consistent_text(sample["prompt"])
            assert code_eval(code, sample["expected"])
