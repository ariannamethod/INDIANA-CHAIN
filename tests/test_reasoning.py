"""Tests for reasoning-related generation helpers."""

from __future__ import annotations

from unittest.mock import patch

from indiana_c.generation import generate_consistent_text, generate_with_think


def test_generate_with_think_returns_thought_and_final() -> None:
    """Ensure ``generate_with_think`` yields both text and metadata."""

    with patch("indiana_c.generation.generate_text", return_value=("thought", {"c": 1})) as mock_gen:
        result = generate_with_think("prompt")

    # The wrapper should request reasoning metadata and return the tuple as-is
    assert result == ("thought", {"c": 1})
    mock_gen.assert_called_once_with("prompt", max_new_tokens=50, config=None, log_reasoning=True)


def test_consistency_improves_with_multiple_attempts() -> None:
    """Using ``n>1`` should select the most frequent answer."""

    side_effect = ["B", "A", "A"]

    # Single attempt may yield an inconsistent answer
    with patch("indiana_c.generation.generate_with_think", side_effect=side_effect):
        single = generate_consistent_text("prompt", n=1)

    # Multiple attempts should recover the majority answer "A"
    with patch("indiana_c.generation.generate_with_think", side_effect=side_effect):
        multi = generate_consistent_text("prompt", n=3)

    assert single != "A"
    assert multi == "A"
