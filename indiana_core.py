"""Unified core for the Indiana-C reasoning engine.

This module consolidates tokenization, logging, prompt loading and text
generation utilities into a single place so that external interfaces only
need to depend on this file. It mirrors the functionality previously spread
across the ``indiana_c`` package.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import List

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from indiana_c.model import IndianaC, IndianaCConfig
from indiana_c.monitor import SelfMonitor
from indiana_c.quantize import quantize_2bit
from indiana_c.reflection import reflect


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

CORE_PROMPT_PATH = Path(__file__).resolve().parent / "core_prompt.txt"
CORE_PROMPT = CORE_PROMPT_PATH.read_text(encoding="utf-8")


def load_core_prompt() -> str:
    """Return the baked-in core prompt."""

    return CORE_PROMPT


def load_prompt(path: str | Path) -> str:
    """Load an arbitrary prompt from ``path``."""

    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
_tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(special_tokens=["[UNK]"])
_tokenizer.train_from_iterator([CORE_PROMPT], trainer)


class TokenizerWrapper:
    """Light wrapper around ``tokenizers.Tokenizer`` providing torch helpers."""

    def __init__(self, tk: Tokenizer):
        self._tk = tk

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def encode(self, text: str) -> torch.Tensor:
        ids = self._tk.encode(text).ids
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        ids = tokens.squeeze().tolist()
        return self._tk.decode(ids)


tokenizer = TokenizerWrapper(_tokenizer)


# ---------------------------------------------------------------------------
# Thought logging
# ---------------------------------------------------------------------------


@dataclass
class ThoughtLogEntry:
    timestamp: str
    message: str
    complexity: int
    entropy: float


class ThoughtComplexityLogger:
    """Track complexity and entropy of generated thoughts."""

    def __init__(self, log_file: str | Path = "logs/thought_log.jsonl") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logs: List[ThoughtLogEntry] = []

    def log_turn(self, message: str, complexity_scale: int, entropy: float) -> ThoughtLogEntry:
        entry = ThoughtLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            message=message,
            complexity=max(1, min(5, complexity_scale)),
            entropy=float(min(1.0, entropy)),
        )
        self.logs.append(entry)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__) + "\n")
        return entry

    def recent(self, n: int = 7) -> List[ThoughtLogEntry]:
        return self.logs[-n:]


def estimate_complexity_and_entropy(message: str) -> tuple[int, float]:
    complexity = 1
    lowered = message.lower()
    if any(keyword in lowered for keyword in ["why", "paradox", "recursive"]):
        complexity += 2
    if len(message) > 300:
        complexity += 1
    complexity = max(1, min(5, complexity))
    unique_words = len(set(message.split()))
    entropy = min(1.0, unique_words / 40)
    return complexity, entropy


thought_logger = ThoughtComplexityLogger()


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------


def generate_text(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
    *,
    log_reasoning: bool = False,
    use_memory: bool = False,
    memory_limit: int = 3,
    self_reflect: bool = False,
) -> str | tuple[str, dict[str, float | int]]:
    """Generate a completion optionally enriched with past prompts."""

    prompt = prompt or CORE_PROMPT
    config = config or IndianaCConfig()
    monitor = SelfMonitor()
    if use_memory:
        examples = monitor.search(prompt, limit=memory_limit)
        if examples:
            combined = "\n".join(
                f"Prompt: {p}\nOutput: {o}" for p, o in examples
            )
            prompt = f"{combined}\n{prompt}"
    model = IndianaC(config)
    quantize_2bit(model)
    model.eval()
    idx = tokenizer.encode(prompt)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0])
    if self_reflect:
        critique = reflect(prompt, text, max_new_tokens=max_new_tokens, config=config)
        if "good" not in critique.lower():
            revision_prompt = (
                f"{prompt}\nDraft answer: {text}\nCritique: {critique}\nRevised answer:"
            )
            idx = tokenizer.encode(revision_prompt)
            out = model.generate(idx, max_new_tokens=max_new_tokens)
            text = tokenizer.decode(out[0])
    monitor.log(prompt, text)
    complexity, entropy = estimate_complexity_and_entropy(text)
    record = thought_logger.log_turn(text, complexity, entropy)
    if log_reasoning:
        return text, {
            "complexity": record.complexity,
            "entropy": record.entropy,
            "timestamp": record.timestamp,
        }
    return text


def reason_loop(
    prompt: str | None = None,
    *,
    max_steps: int = 5,
    stop_tokens: tuple[str, ...] = ("</think>", "</answer>"),
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
) -> str:
    """Iteratively alternate between ``<think>`` and ``<answer>`` phases."""

    prompt = prompt or CORE_PROMPT
    config = config or IndianaCConfig()
    monitor = SelfMonitor()
    model = IndianaC(config)
    quantize_2bit(model)
    model.eval()
    text = prompt
    final_answer = ""
    for _ in range(max_steps):
        think_prompt = f"{text}\n<think>"
        idx = tokenizer.encode(think_prompt)
        out = model.generate(idx, max_new_tokens=max_new_tokens)
        new_tokens = out[:, idx.shape[1] :]
        thought = tokenizer.decode(new_tokens)
        monitor.log("<think>", thought)
        text = tokenizer.decode(out[0])
        if any(tok in thought for tok in stop_tokens):
            break
        answer_prompt = f"{text}\n<answer>"
        idx = tokenizer.encode(answer_prompt)
        out = model.generate(idx, max_new_tokens=max_new_tokens)
        new_tokens = out[:, idx.shape[1] :]
        final_answer = tokenizer.decode(new_tokens)
        monitor.log("<answer>", final_answer)
        text = tokenizer.decode(out[0])
        if any(tok in final_answer for tok in stop_tokens):
            break
    return final_answer or text


def generate_with_think(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
    **kwargs,
) -> str | tuple[str, dict[str, float | int]]:
    """Generate text while requesting reasoning metadata."""

    return generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        config=config,
        log_reasoning=True,
        **kwargs,
    )


def generate_consistent_text(
    prompt: str | None = None,
    n: int = 5,
    **kwargs,
) -> str:
    """Generate ``n`` completions and return the most frequent answer."""

    prompt = prompt or CORE_PROMPT
    results: list[str] = []
    for _ in range(n):
        output = generate_with_think(prompt, **kwargs)
        final = output[-1] if isinstance(output, tuple) else output
        results.append(final)

    counts = Counter(results)
    most_common_answer, freq = counts.most_common(1)[0]
    tied = [ans for ans, c in counts.items() if c == freq]
    if len(tied) > 1:
        most_common_answer = min(tied, key=len)
    return most_common_answer


__all__ = [
    "tokenizer",
    "generate_text",
    "reason_loop",
    "generate_with_think",
    "generate_consistent_text",
    "load_prompt",
    "load_core_prompt",
    "CORE_PROMPT",
    "ThoughtComplexityLogger",
    "estimate_complexity_and_entropy",
    "thought_logger",
]
