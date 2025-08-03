import os
import numpy as np

from indiana_core import SelfMonitor


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


def test_search_embeddings(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)

    class DummyEmbedder:
        def encode(self, text):
            return (
                np.array([1.0, 0.0], dtype=np.float32)
                if "hello" in text
                else np.array([0.0, 1.0], dtype=np.float32)
            )

    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"), embedder=DummyEmbedder())
        monitor.log("hello world", "out1")
        monitor.log("another message", "out2")
        results = monitor.search("hello there", method="embedding")
        assert results[0] == ("hello world", "out1")
    finally:
        os.chdir(cwd)
