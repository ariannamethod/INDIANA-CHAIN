import os
import hashlib
import time

from indiana_c.monitor import SelfMonitor


def test_search_exact(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))
        monitor.log("hello world", "out1")
        monitor.log("another message", "out2")
        results = monitor.search("hello world")
        assert ("hello world", "out1") in results
    finally:
        os.chdir(cwd)


def test_search_tfidf_limit(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))
        for i in range(3):
            monitor.log(f"hello {i}", f"out{i}")
        results = monitor.search("hello", limit=2)
        assert len(results) == 2
        assert all("hello" in p for p, _ in results)
    finally:
        os.chdir(cwd)


def test_dataset_watching(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ds = tmp_path / "datasets"
        ds.mkdir()
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"), watch_dirs=[ds])
        file_path = ds / "sample.txt"
        file_path.write_text("hello", encoding="utf-8")
        time.sleep(0.2)
        cur = monitor.conn.cursor()
        cur.execute("SELECT sha256 FROM files WHERE path = ?", (str(file_path),))
        row = cur.fetchone()
        assert row is not None
        file_path.write_text("hello world", encoding="utf-8")
        time.sleep(0.2)
        cur.execute("SELECT sha256 FROM files WHERE path = ?", (str(file_path),))
        sha = hashlib.sha256(b"hello world").hexdigest()
        assert cur.fetchone()[0] == sha
        monitor.close()
    finally:
        os.chdir(cwd)
