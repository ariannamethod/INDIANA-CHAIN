import os

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


def test_max_logs_limit(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"), max_logs=5)
        for i in range(10):
            monitor.log(f"prompt {i}", f"out{i}")
        cur = monitor.conn.cursor()
        cur.execute("SELECT prompt FROM logs ORDER BY ts")
        prompts = [row[0] for row in cur.fetchall()]
        assert len(prompts) == 5
        assert prompts == [f"prompt {i}" for i in range(5, 10)]
        cur.execute("SELECT COUNT(*) FROM prompts_index")
        assert cur.fetchone()[0] == 5
    finally:
        os.chdir(cwd)
