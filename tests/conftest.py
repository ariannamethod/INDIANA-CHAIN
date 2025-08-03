import pytest

from indiana_core import MODEL_CACHE


@pytest.fixture(autouse=True)
def clear_model_cache():
    MODEL_CACHE.clear()
    yield
    MODEL_CACHE.clear()
