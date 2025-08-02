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
