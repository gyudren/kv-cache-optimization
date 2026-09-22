"""Single GPT-5.6 Sol structured-output channel for Generator and Judge.

A client with an OpenAI Responses API ``responses.parse`` method is required.
No invented mock results are emitted on API failures.
"""
from __future__ import annotations
from typing import TypeVar
from pydantic import BaseModel
from .config import MODEL_ID
T = TypeVar("T", bound=BaseModel)

class StructuredLLM:
    def __init__(self, api_key: str, client: object | None = None):
        if not api_key and client is None:
            raise ValueError("OPENAI_API_KEY is required")
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = MODEL_ID

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[{"role": "system", "content": "Use only provided evidence. If unsupported, report missing information. Preserve verified citations; no fabricated sources."},
                       {"role": "user", "content": prompt}],
                text_format=schema,
            )
        except AttributeError as exc:
            raise RuntimeError("Install a recent OpenAI SDK with responses.parse support") from exc
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("GPT-5.6 Sol returned no parseable structured output")
        return parsed
