"""Self-monitoring utilities for Indiana-C.

The monitor snapshot the entire repository and logs prompts and
responses for further fine-tuning. All files are stored inside an
SQLite database so the system can reflect on its own source and data.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers.polling import PollingObserver


class _FileEventHandler(FileSystemEventHandler):
    """Internal handler that updates the database on file events."""

    def __init__(self, monitor: "SelfMonitor") -> None:
        self.monitor = monitor

    def on_created(self, event: FileSystemEvent) -> None:  # pragma: no cover - thin wrapper
        if not event.is_directory:
            self.monitor.snapshot_file(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:  # pragma: no cover - thin wrapper
        if not event.is_directory:
            self.monitor.snapshot_file(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:  # pragma: no cover - thin wrapper
        if not event.is_directory:
            self.monitor.remove_file(event.src_path)
            self.monitor.snapshot_file(event.dest_path)

    def on_deleted(self, event: FileSystemEvent) -> None:  # pragma: no cover - thin wrapper
        if not event.is_directory:
            self.monitor.remove_file(event.src_path)


class SelfMonitor:
    """Record code snapshots and generation events."""

    def __init__(
        self,
        db_path: str = "indiana_memory.sqlite",
        watch_dirs: Iterable[str | Path] | None = ("datasets",),
        poll_interval: float = 0.1,
    ) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()
        self.snapshot_codebase()
        self._observer: PollingObserver | None = None
        if watch_dirs:
            self._start_watching(watch_dirs, poll_interval)

    def _init_db(self) -> None:
        with self.lock:
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

    def snapshot_codebase(self, root: str | Path = ".") -> None:
        """Store all files in the repository with their hashes."""
        root_path = Path(root)
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "indiana_memory.sqlite":
                continue
            self.snapshot_file(path)

    def snapshot_file(self, path: str | Path) -> None:
        path = Path(path)
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

    def remove_file(self, path: str | Path) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM files WHERE path = ?", (str(Path(path)),))
            self.conn.commit()

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

    def _start_watching(
        self, watch_dirs: Iterable[str | Path], poll_interval: float
    ) -> None:
        handler = _FileEventHandler(self)
        observer = PollingObserver(timeout=poll_interval)
        for d in watch_dirs:
            path = Path(d)
            if path.exists():
                observer.schedule(handler, str(path), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer

    def close(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        self.conn.close()

    def __del__(self) -> None:  # pragma: no cover - cleanup helper
        try:
            self.close()
        except Exception:
            pass


__all__ = ["SelfMonitor"]
