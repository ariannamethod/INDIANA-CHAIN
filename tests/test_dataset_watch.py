import os
import time
from pathlib import Path

from indiana_core import SelfMonitor


def test_dataset_watch(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    monitor = None
    try:
        ds = Path("datasets")
        ds.mkdir()
        monitor = SelfMonitor(db_path=str(tmp_path / "mem.sqlite"))
        file = ds / "sample.txt"
        file.write_text("hello")
        time.sleep(0.5)
        cur = monitor.conn.cursor()
        cur.execute("SELECT content FROM files WHERE path = ?", (str(file),))
        row = cur.fetchone()
        assert row is not None
    finally:
        if monitor:
            monitor.stop_watchers()
        os.chdir(cwd)
