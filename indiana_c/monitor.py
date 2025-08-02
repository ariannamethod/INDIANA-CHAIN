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

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import threading


class _SnapshotHandler(FileSystemEventHandler):
    """Watchdog handler that snapshots changed files."""

    def __init__(self, monitor: "SelfMonitor"):
        self.monitor = monitor

    def on_modified(self, event):  # type: ignore[override]
        if not event.is_directory:
            self.monitor._snapshot_file(Path(event.src_path))

    # Trigger same snapshot for created and moved events
    on_created = on_modified  # type: ignore[assignment]
    on_moved = on_modified  # type: ignore[assignment]


class SelfMonitor:
    """Record code snapshots and generation events."""

    def __init__(
        self,
        db_path: str = "indiana_memory.sqlite",
        *,
        watch_datasets: bool = True,
    ):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()
        self.observers: dict[str, Observer] = {}
        self.snapshot_codebase()

        if watch_datasets:
            datasets_dir = Path("datasets")
            if datasets_dir.exists():
                self.watch_directory(datasets_dir)

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, content BLOB, sha256 TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS logs(ts REAL, prompt TEXT, output TEXT, sha256 TEXT)"
        )
        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS prompts_index USING fts5(prompt, output)"
        )
        self.conn.commit()

    def _snapshot_file(self, path: Path) -> None:
        """Snapshot a single file into the database."""
        if not path.is_file() or path.name == "indiana_memory.sqlite":
            return
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO files(path, content, sha256) VALUES (?,?,?)",
                (str(path), sqlite3.Binary(data), sha),
            )
            self.conn.commit()

    def snapshot_codebase(self, root: str | Path = ".") -> None:
        """Store all files in the repository with their hashes."""
        root_path = Path(root)
        if root_path.is_file():
            self._snapshot_file(root_path)
            return
        for path in root_path.rglob("*"):
            self._snapshot_file(path)

    def log(self, prompt: str, output: str) -> None:
        """Log a generation event with timestamp."""
        sha = hashlib.sha256(prompt.encode()).hexdigest()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO logs(ts, prompt, output, sha256) VALUES (?,?,?,?)",
                (time.time(), prompt, output, sha),
            )
            cur.execute(
                "INSERT INTO prompts_index(prompt, output) VALUES (?,?)",
                (prompt, output),
            )
            self.conn.commit()

    def _search_tfidf(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT prompt, output FROM prompts_index WHERE prompts_index MATCH ? "
                "ORDER BY bm25(prompts_index) LIMIT ?",
                (query, limit),
            )
            return cur.fetchall()

    def search(self, prompt: str, limit: int = 5) -> list[tuple[str, str]]:
        """Return top-k similar prompt/output pairs.

        Exact SHA-256 matches are preferred; otherwise a TF-IDF lookup is used.
        """

        sha = hashlib.sha256(prompt.encode()).hexdigest()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT prompt, output FROM logs WHERE sha256 = ? LIMIT ?",
                (sha, limit),
            )
            rows = cur.fetchall()
        if rows:
            return rows
        return self._search_tfidf(prompt, limit=limit)

    def search_prompts(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """Search previously logged prompts similar to the query."""
        return self._search_tfidf(query, limit=limit)

    def watch_directory(self, path: str | Path) -> None:
        """Begin watching a directory for changes."""
        path = str(Path(path))
        if path in self.observers:
            return
        handler = _SnapshotHandler(self)
        observer = Observer()
        observer.schedule(handler, path, recursive=True)
        observer.daemon = True
        observer.start()
        self.observers[path] = observer

    def stop_watchers(self) -> None:
        """Stop all active directory watchers."""
        for observer in self.observers.values():
            observer.stop()
            observer.join()
        self.observers.clear()


__all__ = ["SelfMonitor"]
