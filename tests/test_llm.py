from types import SimpleNamespace
from pydantic import BaseModel
from kv_eval.llm import StructuredLLM
from kv_eval.config import Settings, MODEL_ID


class Output(BaseModel):
    message: str


class FakeResponses:
    def __init__(self):
        self.params = []
    def parse(self, **kwargs):
        self.params.append(kwargs)
        return SimpleNamespace(output_parsed=Output(message="test"))


def test_fixed_model_and_structured_output():
    stub = FakeResponses()
    llm = StructuredLLM("", client=SimpleNamespace(responses=stub))
    out = llm.generate_structured("prompt", Output)
    assert out.message == "test"
    # 대체 모델 없이 설정된 단일 모델만 호출해야 한다.
    assert stub.params[0]["model"] == MODEL_ID


def test_model_override_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "other-model")
    import pytest
    with pytest.raises(ValueError):
        Settings.from_env()
