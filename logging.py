import sys
import importlib.machinery as _machinery
import importlib.util as _util
from collections import Counter
import math

# Load the standard library logging module under a different spec
_spec = _machinery.PathFinder.find_spec("logging", sys.path[1:])
if _spec and _spec.loader:
    _stdlib_logging = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_stdlib_logging)
    for _name in dir(_stdlib_logging):
        if _name.startswith("__"):
            continue
        globals()[_name] = getattr(_stdlib_logging, _name)


def lexical_entropy(text: str, n: int = 2) -> float:
    """Compute normalized lexical entropy based on n-gram frequencies."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy if max_entropy > 0 else 0.0)


def simple_perplexity(text: str, tokenizer, model) -> float:
    """Estimate perplexity of ``text`` using ``tokenizer`` and ``model``.

    The model is expected to behave like ``IndianaC`` and return logits and
    cross-entropy loss when called with ``(ids[:, :-1], ids[:, 1:])``.
    """
    import torch

    with torch.no_grad():
        ids = tokenizer.encode(text)
        if ids.size(1) < 2:
            return float("inf")
        logits, loss = model(ids[:, :-1], ids[:, 1:])
        if loss is None:
            loss = torch.tensor(0.0)
    return float(torch.exp(loss).item())
