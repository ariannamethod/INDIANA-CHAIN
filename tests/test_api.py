from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import app


def test_generate_endpoint_returns_response() -> None:
    client = TestClient(app)
    with patch("app.generate_text", return_value="ok"):
        resp = client.post("/generate", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json() == {"response": "ok"}


@pytest.mark.parametrize("payload", [{}, {"prompt": ""}])
def test_generate_empty_prompt(payload) -> None:
    client = TestClient(app)
    with patch("app.generate_text", return_value="ok"):
        resp = client.post("/generate", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"response": "ok"}
