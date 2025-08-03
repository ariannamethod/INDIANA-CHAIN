from indiana_core import (
    estimate_complexity_and_entropy,
    ThoughtComplexityLogger,
)


def test_estimate_complexity_and_entropy_keywords():
    msg = "This is a paradox that asks why it is recursive"
    complexity, entropy, steps = estimate_complexity_and_entropy(msg)
    assert complexity >= 3
    assert 0 <= entropy <= 1
    assert steps == 0


def test_estimate_complexity_and_entropy_counts_steps():
    msg = "Step 1: analyze\nStep 2: conclude"
    _, _, steps = estimate_complexity_and_entropy(msg)
    assert steps == 2


def test_logger_records_and_recent():
    logger = ThoughtComplexityLogger(log_file="logs/test_log.jsonl")
    entry = logger.log_turn("test message", 2, 0.5, 3)
    assert entry.complexity == 2
    assert entry.steps == 3
    assert logger.recent(1)[0].message == "test message"
