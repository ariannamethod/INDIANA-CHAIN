from unittest.mock import patch

from indiana_core import generate_consistent_text


def test_generate_consistent_text_majority():
    with patch(
        "indiana_core.generate_with_think", side_effect=["A", "A", "B"]
    ) as mocked:
        result = generate_consistent_text("prompt", n=3)
        assert result == "A"
        assert mocked.call_count == 3
