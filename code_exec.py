"""Utility module to execute and verify Python code snippets.

This module provides helper functions for sandboxed execution of small Python
snippets and simple verification hooks that can be used in unit tests or to
check solutions against sample inputs. It is inspired by the competitive
programming utilities typically used to run submissions against predefined
test cases.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Tuple


def run_python_snippet(code: str, stdin: str | None = None, timeout: float = 5.0) -> Tuple[str, str, int]:
    """Execute ``code`` in a separate Python process.

    Parameters
    ----------
    code:
        The Python source to execute.
    stdin:
        Optional standard input to feed the process.
    timeout:
        Maximum time in seconds to allow the process to run.

    Returns
    -------
    tuple
        A tuple ``(stdout, stderr, returncode)`` capturing the result of the
        execution.
    """

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(code)
        tmp_path = handle.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        os.remove(tmp_path)


def evaluate(code: str, tests: Iterable[Tuple[str, str]], timeout: float = 5.0) -> dict[str, Any]:
    """Run ``code`` against a series of ``tests`` and compute a reward.

    Each element in ``tests`` is a ``(input, expected_output)`` tuple. For each
    case the snippet is executed and its output is compared to the expected
    string. The resulting dictionary contains a per-test breakdown along with
    an aggregate reward between 0 and 1, representing the fraction of passing
    tests.
    """

    results = []
    for inp, expected in tests:
        stdout, stderr, returncode = run_python_snippet(code, inp, timeout)
        passed = stdout.strip() == expected.strip() and not stderr and returncode == 0
        results.append(
            {
                "input": inp,
                "expected": expected,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "passed": passed,
            }
        )

    reward = sum(r["passed"] for r in results) / len(results) if results else 0.0
    return {"passed": all(r["passed"] for r in results), "reward": reward, "results": results}


__all__ = ["run_python_snippet", "evaluate"]
