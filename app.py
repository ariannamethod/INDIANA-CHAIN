"""Simple FastAPI interface exposing the Indiana Chain core."""

from fastapi import FastAPI
from pydantic import BaseModel

from indiana_core import generate_text


app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str | None = None


class GenerateResponse(BaseModel):
    response: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a completion for ``req.prompt``."""

    text = generate_text(req.prompt)
    return GenerateResponse(response=text)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
