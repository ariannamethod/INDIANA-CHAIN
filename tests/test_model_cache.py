from unittest.mock import patch

from indiana_core import IndianaCConfig, generate_text, tokenizer


def test_model_cache_reuses_model() -> None:
    config = IndianaCConfig()
    with patch("indiana_core.IndianaC") as MockModel, patch(
        "indiana_core.quantize_2bit"
    ) as mock_quant:
        instance = MockModel.return_value
        instance.generate.return_value = tokenizer.encode("ok")
        instance.eval.return_value = None
        generate_text("prompt", config=config)
        generate_text("prompt", config=config)
    assert MockModel.call_count == 1
    assert mock_quant.call_count == 1
