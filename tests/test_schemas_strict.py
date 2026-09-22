"""OpenAI 구조화 출력(strict) 제약을 로컬에서 검증한다.

strict 스키마는 자유형 object(`dict[str, ...]`)를 허용하지 않는다. 이를 어기면 네트워크
호출 시점에야 400 invalid_json_schema 로 터지므로(실제로 파이프라인이 여기서 죽었다),
스키마 정의만 보고 미리 걸러낸다.
"""
import pytest
from pydantic import BaseModel

from kv_eval import schemas


def _models() -> list[type[BaseModel]]:
    return [obj for obj in vars(schemas).values()
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel]


def _free_form_objects(schema: dict, path: str = "") -> list[str]:
    """properties 없이 임의 키를 받는 object 노드(=자유형 dict)를 찾는다."""
    found: list[str] = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" not in schema:
            found.append(path or "<root>")
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                for name, sub in value.items():
                    found += _free_form_objects(sub, f"{path}.{name}" if path else name)
            elif isinstance(value, dict):
                found += _free_form_objects(value, path)
            elif isinstance(value, list):
                for item in value:
                    found += _free_form_objects(item, path)
    return found


@pytest.mark.parametrize("model", _models(), ids=lambda m: m.__name__)
def test_schema_has_no_free_form_dict(model: type[BaseModel]):
    offenders = _free_form_objects(model.model_json_schema())
    assert not offenders, (
        f"{model.__name__}의 {offenders} 필드가 자유형 dict입니다. "
        "OpenAI strict 스키마가 거부하므로 키를 명시한 하위 모델로 바꾸세요."
    )
