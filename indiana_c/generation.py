from collections import Counter
from pathlib import Path

from .model import IndianaC, IndianaCConfig
from .monitor import SelfMonitor
from .quantize import quantize
from .logger import (
    estimate_complexity_and_entropy,
    thought_logger,
)
from .tokenizer import tokenizer
from .reflection import reflect

CORE_PROMPT = (
    Path(__file__).resolve().parent.parent / "core_prompt.txt"
).read_text(encoding="utf-8")
print("core_prompt.txt loaded [OK]")


def generate_text(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
    *,
    log_reasoning: bool = False,
    use_history: bool = False,
    history_limit: int = 3,
    self_reflect: bool = False,
) -> str | tuple[str, dict[str, float | int]]:
    """Generate a completion optionally enriched with past prompts.

    Args:
        prompt: Initial text to complete. If ``None`` the core prompt is used.
        max_new_tokens: Maximum number of tokens to generate.
        config: Optional model configuration.
        log_reasoning: Whether to return reasoning metadata.
        use_history: Fetch similar past prompts from :class:`SelfMonitor` and
            prepend them to the provided prompt.
        history_limit: Maximum number of historical prompts to include.

    Returns:
        The generated text. If ``log_reasoning`` is ``True`` a tuple of the text
        and a dictionary with reasoning statistics is returned instead.
    """
    prompt = prompt or CORE_PROMPT
    config = config or IndianaCConfig()
    monitor = SelfMonitor()
    if use_history:
        history = monitor.search_prompts(prompt, limit=history_limit)
        if history:
            combined = "\n".join(p for p, _ in history)
            prompt = f"{combined}\n{prompt}"
    model = IndianaC(config)
    quantize(model, config.quantization_bits)
    model.eval()
    idx = tokenizer.encode(prompt)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0])
    if self_reflect:
        critique = reflect(prompt, text, max_new_tokens=max_new_tokens, config=config)
        if "good" not in critique.lower():
            revision_prompt = (
                f"{prompt}\nDraft answer: {text}\nCritique: {critique}\nRevised answer:"
            )
            idx = tokenizer.encode(revision_prompt)
            out = model.generate(idx, max_new_tokens=max_new_tokens)
            text = tokenizer.decode(out[0])
    monitor.log(prompt, text)
    complexity, entropy = estimate_complexity_and_entropy(text)
    record = thought_logger.log_turn(text, complexity, entropy)
    if log_reasoning:
        return text, {
            "complexity": record.complexity,
            "entropy": record.entropy,
            "timestamp": record.timestamp,
        }
    return text


def generate_with_think(
    prompt: str | None = None,
    max_new_tokens: int = 50,
    config: IndianaCConfig | None = None,
    **kwargs,
) -> str | tuple[str, dict[str, float | int]]:
    """Generate text while allowing a hook for reasoning steps.

    Currently this is a light wrapper around :func:`generate_text` so that it can
    be mocked in tests and extended in the future. The function requests
    reasoning metadata from :func:`generate_text` and therefore returns a tuple
    of the generated text and the associated statistics.
    """

    return generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        config=config,
        log_reasoning=True,
        **kwargs,
    )


def generate_consistent_text(
    prompt: str | None = None,
    n: int = 5,
    **kwargs,
) -> str:
    """Generate multiple completions and return the most consistent answer.

    Args:
        prompt: Optional prompt to complete. If ``None`` the core prompt is used.
        n: Number of attempts to generate a completion.
        **kwargs: Extra arguments passed to :func:`generate_with_think`.

    Returns:
        The most frequently produced final answer. In case of a tie, the
        shortest answer is returned.
    """

    prompt = prompt or CORE_PROMPT
    results: list[str] = []
    for _ in range(n):
        output = generate_with_think(prompt, **kwargs)
        final = output[-1] if isinstance(output, tuple) else output
        results.append(final)

    counts = Counter(results)
    most_common_answer, freq = counts.most_common(1)[0]
    tied = [ans for ans, c in counts.items() if c == freq]
    if len(tied) > 1:
        most_common_answer = min(tied, key=len)
    return most_common_answer
