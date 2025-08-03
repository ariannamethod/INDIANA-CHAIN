"""Utility for safe execution of generated code snippets."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any, Dict

# A minimal set of safe built-in functions that user code may rely on.
_SAFE_BUILTINS = {
    "abs": abs,
    "enumerate": enumerate,
    "len": len,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "sum": sum,
}


def code_eval(code: str, expected: Any) -> bool:
    """Execute ``code`` in a restricted environment and compare ``result`` with ``expected``.

    The executed snippet should assign the answer to a variable named ``result``. The function
    returns ``True`` if the variable equals ``expected`` and ``False`` otherwise. Any exception
    during execution is caught and treated as a failure.
    """

    globals_dict: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    locals_dict: Dict[str, Any] = {}
    stdout = io.StringIO()

    try:
        with redirect_stdout(stdout):
            exec(code, globals_dict, locals_dict)
    except Exception:
        return False

    if "result" in locals_dict:
        actual = locals_dict["result"]
    else:
        actual = stdout.getvalue().strip()

    return actual == expected
