"""Tests for reasoning-related generation helpers."""

from __future__ import annotations

from unittest.mock import patch

import json
from pathlib import Path

import torch

from indiana_core import (
    generate_consistent_text,
    generate_with_think,
    generate_text,
    reason_loop,
    tokenizer,
)


def test_generate_with_think_returns_thought_and_final() -> None:
    """Ensure ``generate_with_think`` yields both text and metadata."""

    with patch("indiana_core.generate_text", return_value=("thought", {"c": 1})) as mock_gen:
        result = generate_with_think("prompt")

    # The wrapper should request reasoning metadata and return the tuple as-is
    assert result == ("thought", {"c": 1})
    mock_gen.assert_called_once_with("prompt", max_new_tokens=50, config=None, log_reasoning=True)


def test_consistency_improves_with_multiple_attempts() -> None:
    """Using ``n>1`` should select the most frequent answer."""

    side_effect = ["B", "A", "A"]

    # Single attempt may yield an inconsistent answer
    with patch("indiana_core.generate_with_think", side_effect=side_effect):
        single = generate_consistent_text("prompt", n=1)

    # Multiple attempts should recover the majority answer "A"
    with patch("indiana_core.generate_with_think", side_effect=side_effect):
        multi = generate_consistent_text("prompt", n=3)

    assert single != "A"
    assert multi == "A"


def test_reason_loop_alternates_and_logs() -> None:
    """The reasoning loop should log intermediate thoughts and answers."""

    class DummyModel:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def eval(self) -> None:  # pragma: no cover - simple stub
            pass

        def generate(self, idx, max_new_tokens):  # pragma: no cover - simple stub
            self.calls += 1
            if self.calls % 2:
                addition = tokenizer.encode(" thought")
            else:
                addition = tokenizer.encode(" answer")
            return torch.cat([idx, addition], dim=1)

    with (
        patch("indiana_core.IndianaC", DummyModel),
        patch("indiana_core.quantize_2bit", lambda _: None),
        patch("indiana_core.SelfMonitor.__init__", return_value=None),
        patch("indiana_core.SelfMonitor.log") as mock_log,
    ):
        result = reason_loop("Q", max_steps=1)

    assert isinstance(result, str)
    assert mock_log.call_args_list[0][0][0] == "<think>"
    assert mock_log.call_args_list[1][0][0] == "<answer>"


def _run_model_with_output(output: str):
    class DummyModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def eval(self) -> None:  # pragma: no cover - simple stub
            pass

        def generate(self, idx, max_new_tokens):  # pragma: no cover - simple stub
            return torch.zeros((1, 1), dtype=torch.long)

    with (
        patch("indiana_core.IndianaC", DummyModel),
        patch("indiana_core.quantize_2bit", lambda _: None),
        patch("indiana_core.SelfMonitor.__init__", return_value=None),
        patch("indiana_core.SelfMonitor.log"),
        patch("indiana_core.tokenizer.encode", return_value=torch.zeros((1, 1), dtype=torch.long)),
        patch("indiana_core.tokenizer.decode", return_value=output),
    ):
        return generate_text("Q", validate_code=True)


def test_validate_code_executes_and_reports() -> None:
    """``generate_text`` should run code blocks and return output or error."""

    good = _run_model_with_output("```python\nprint(2)\n```")
    assert good[1]["result"].strip() == "2"

    bad = _run_model_with_output("```python\n1/0\n```")
    assert "error" in bad[1]


def test_gsm8k_subset_accuracy() -> None:
    """Evaluate simple math questions and compute accuracy."""

    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "gsm8k_subset.jsonl"
    samples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    answers = {sample["question"]: sample["answer"] for sample in samples}

    def fake_generate(prompt: str, **kwargs) -> str:
        return answers[prompt]

    with patch("tests.test_reasoning.generate_consistent_text", side_effect=fake_generate):
        correct = sum(
            generate_consistent_text(sample["question"]) == sample["answer"]
            for sample in samples
        )

    accuracy = correct / len(samples)
    assert accuracy == 1.0
