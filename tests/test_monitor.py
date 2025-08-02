import os

from indiana_c.monitor import SelfMonitor


def test_search_prompts(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))
        monitor.log("hello world", "out1")
        monitor.log("another message", "out2")
        results = monitor.search_prompts("hello")
        assert ("hello world", "out1") in results
    finally:
        os.chdir(cwd)


def test_search_prompts_limit(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))
        for i in range(3):
            monitor.log(f"hello {i}", f"out{i}")
        results = monitor.search_prompts("hello", limit=2)
        assert len(results) == 2
    finally:
        os.chdir(cwd)
