from types import SimpleNamespace
from unittest.mock import patch

from indiana_core import generate_text, tokenizer


class DummyMonitor:
    def __init__(self, *_, **__):
        pass

    def search_prompts(self, *_args, **_kwargs):
        return []

    def log(self, *_args, **_kwargs):
        pass


def _patch_env():
    return (
        patch("indiana_core.SelfMonitor", DummyMonitor),
        patch("indiana_core.quantize_2bit"),
        patch(
            "indiana_core.thought_logger.log_turn",
            return_value=SimpleNamespace(complexity=1, entropy=0.1, timestamp="t"),
        ),
    )


def test_reflection_regenerates_when_reward_low() -> None:
    draft = tokenizer.encode("draft")
    revised1 = tokenizer.encode("revised1")
    revised2 = tokenizer.encode("revised2")
    p1, p2, p3 = _patch_env()
    with (
        p1,
        p2,
        p3,
        patch("indiana_core.reflect", return_value="critique"),
        patch("indiana_core.accuracy_reward", return_value=0.0),
        patch("indiana_core.IndianaC") as MockModel,
    ):
        mock = MockModel.return_value
        mock.generate.side_effect = [draft, revised1, revised2]
        mock.eval.return_value = None
        result = generate_text("prompt", self_reflect=True)
    assert result == tokenizer.decode(revised2)
    assert mock.generate.call_count == 3


def test_reflection_accepts_revised_when_reward_high() -> None:
    draft = tokenizer.encode("draft")
    revised = tokenizer.encode("revised")
    p1, p2, p3 = _patch_env()
    with (
        p1,
        p2,
        p3,
        patch("indiana_core.reflect", return_value="critique"),
        patch("indiana_core.accuracy_reward", return_value=1.0),
        patch("indiana_core.IndianaC") as MockModel,
    ):
        mock = MockModel.return_value
        mock.generate.side_effect = [draft, revised]
        mock.eval.return_value = None
        result = generate_text("prompt", self_reflect=True)
    assert result == tokenizer.decode(revised)
    assert mock.generate.call_count == 2
