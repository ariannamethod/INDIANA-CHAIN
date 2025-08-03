import atexit
import os
import time
from pathlib import Path

import indiana_core


def test_monitor_stops_watchers_on_atexit(tmp_path, monkeypatch):
    callbacks = []

    def fake_register(func, *args, **kwargs):
        callbacks.append((func, args, kwargs))
        return func

    monkeypatch.setattr(atexit, "register", fake_register)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    ds = Path("datasets")
    ds.mkdir()
    try:
        monitor = indiana_core.get_monitor()
        time.sleep(0.1)
        assert monitor.observers
        assert all(obs.is_alive() for obs in monitor.observers.values())
        assert len(callbacks) == 1
        for func, args, kwargs in callbacks:
            func(*args, **kwargs)
        assert not monitor.observers
    finally:
        indiana_core._monitor_instance = None
        os.chdir(cwd)
