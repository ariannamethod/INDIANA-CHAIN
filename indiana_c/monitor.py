"""Self-monitoring utilities for Indiana-C.

The monitor snapshot the entire repository and logs prompts and
responses for further fine-tuning. All files are stored inside an
SQLite database so the system can reflect on its own source and data.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


class SelfMonitor:
    """Record code snapshots and generation events."""

    def __init__(self, db_path: str = "indiana_memory.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.snapshot_codebase()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, content BLOB, sha256 TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS logs(ts REAL, prompt TEXT, output TEXT)"
        )
        self.conn.commit()

    def snapshot_codebase(self, root: str | Path = ".") -> None:
        """Store all files in the repository with their hashes."""
        root_path = Path(root)
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "indiana_memory.sqlite":
                continue
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO files(path, content, sha256) VALUES (?,?,?)",
                (str(path), sqlite3.Binary(data), sha),
            )
        self.conn.commit()

    def log(self, prompt: str, output: str) -> None:
        """Log a generation event with timestamp."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO logs(ts, prompt, output) VALUES (?,?,?)",
            (time.time(), prompt, output),
        )
        self.conn.commit()


__all__ = ["SelfMonitor"]
