from kv_eval.prompts import prompt_template


def test_role_prompt_uses_common_contract_and_new_role_file():
    prompt = prompt_template("technology")

    assert "# COMMON CONTRACT" in prompt
    assert "너는 LLM 추론, attention, KV cache" in prompt
    assert "# 기술 조사 Agent — 요구사항" not in prompt
    assert "## TASK TEMPLATE" not in prompt


def test_master_gate_prompt_combines_master_and_validator_once():
    prompt = prompt_template("master", "validator")

    assert prompt.count("# COMMON CONTRACT") == 1
    assert "너는 KV cache 최적화 기술 평가 파이프라인의 총괄 오케스트레이터다" in prompt
    assert "너는 Multi-Agent 평가 결과의 형식·근거·충분성을 검사하는 품질 검증자다" in prompt


def test_unknown_prompt_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="Unsupported design prompt"):
        prompt_template("legacy")
