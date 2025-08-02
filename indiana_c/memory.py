"""Simple logging utilities for Indiana-C."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parent.parent / "datasets" / "self_log.jsonl"


def log_interaction(prompt: str, response: str) -> None:
    """Append a prompt/response pair to the self training log."""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.utcnow().isoformat(), "prompt": prompt, "response": response}
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
