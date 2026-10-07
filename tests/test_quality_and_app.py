"""품질 평가 규칙, 보고서 규격(E13), 모듈 분리(E1), CLI 진입점 검증."""
from __future__ import annotations
import importlib.util
import json
import re
from pathlib import Path

import pytest

from fakes import FakeLLM
from kv_eval.config import MAX_REPORT_PAGES, REPORT_STEM
from kv_eval.evaluation.quality import check_bias, check_coverage, check_groundedness, check_neutrality

ROOT = Path(__file__).resolve().parents[1]


def test_neutrality_rule_ignores_disclaimers():
    ok = {"report": "## SUMMARY\n본 보고서는 순위, 최종 승자 또는 단일 추천을 제시하지 않는다.\n## REFERENCE\n"}
    bad = {"report": "## SUMMARY\nITME가 MLA보다 우월하다.\n## REFERENCE\n"}
    assert check_neutrality(ok)["passed"] is True
    result = check_neutrality(bad)
    assert result["passed"] is False and result["targets"] == ["report"]


def test_groundedness_requires_citation_on_quantitative_sentence(run_graph):
    state = run_graph(FakeLLM())
    assert check_groundedness(state)["passed"] is True
    tampered = {**state, "report": state["report"].replace("## 6. 시사점\n", "## 6. 시사점\nKV cache가 93.3% 감소한다.\n", 1)}
    result = check_groundedness(tampered)
    assert result["passed"] is False and any("수치 문장" in i for i in result["issues"])


def test_bias_rule_flags_single_source_and_one_sided_verdicts(run_graph):
    state = run_graph(FakeLLM())
    web = [ev for ev in state["evidence"] if ev.get("agent") == "market" and ev.get("technology") == "mla"]
    skewed = [ev for ev in state["evidence"] if ev not in web] + [{**ev, "publisher": "one.example.com"} for ev in web]
    result = check_bias({**state, "evidence": skewed})
    assert result["passed"] is False and result["targets"] == ["market"]
    assert any("단일 발행처" in i for i in result["issues"])
    # Supervisor가 공백으로 기록한 관점·기술은 '근거 부족 명시'로 인정한다
    assert check_bias({**state, "evidence": skewed, "gaps": ["market/mla: 재조사 한도 소진"]})["passed"] is True


def test_coverage_rule_targets_missing_perspective(run_graph):
    state = run_graph(FakeLLM())
    report = re.sub(r"### 4\.3 이해관계자\n.*?(?=### 4\.4)", "### 4.3 이해관계자\n\n", state["report"], flags=re.S)
    result = check_coverage({**state, "report": report})
    assert result["passed"] is False and result["targets"] == ["stakeholder"]


def test_agents_do_not_import_supervisor():
    for path in (ROOT / "src" / "kv_eval" / "agents").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        modules = re.findall(r"^\s*(?:from|import)\s+(\S+)", text, re.M)
        assert not any("supervisor" in module for module in modules), path.name


def test_quality_prompt_registered():
    from kv_eval.prompts import prompt_template
    text = prompt_template("quality_evaluator")
    assert "groundedness" in text and "coverage" in text


def _load_app():
    spec = importlib.util.spec_from_file_location("kv_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_cli_and_finalize_exports_agent_report(run_graph, isolated_outputs):
    from kv_eval.reporting.export import korean_fonts
    try:
        korean_fonts()
    except RuntimeError:
        pytest.skip("한글 TrueType 폰트가 없어 PDF를 만들 수 없는 환경")
    app = _load_app()
    assert app.parse_args(["--resume", "abc"]).resume == "abc"
    state = run_graph(FakeLLM())
    out = isolated_outputs / "outputs"
    result = app.save_and_finalize(state, out)
    assert result["verified"] is True and result["trace_id"] == state["trace_id"]
    assert Path(result["pdf"]).name == f"{REPORT_STEM}.pdf" and REPORT_STEM.startswith("Agent_판교_9반_")
    assert 1 <= result["pages"] <= MAX_REPORT_PAGES
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    assert validation["pdf_pages"] == result["pages"] and set(validation["quality_eval"]) == {
        "groundedness", "neutrality", "bias_control", "coverage"}
    logs = json.loads((out / "run_logs.json").read_text(encoding="utf-8"))
    assert logs and logs[0]["trace_id"] == state["trace_id"]


def test_page_limit_is_enforced():
    from kv_eval.reporting.export import korean_fonts
    from kv_eval.reporting.sections import validate_report
    try:
        korean_fonts()
    except RuntimeError:
        pytest.skip("한글 TrueType 폰트 없음")
    filler = "\n\n".join("근거 부족 문단 " * 60 for _ in range(220))
    headings = ["SUMMARY", "1. 분석 배경", "2. 기술 선정", "3. 기술 개요", "4. 관점별 평가", "5. 종합 의견",
                "6. 시사점", "7. 한계점", "REFERENCE"]
    report = "\n".join(f"## {h}\n{'요약' if h == 'SUMMARY' else filler if h == '3. 기술 개요' else '근거 부족'}"
                       for h in headings) + "\n"
    result = validate_report(report, [])
    assert result["pdf_pages"] > MAX_REPORT_PAGES
    assert any(f"최대 {MAX_REPORT_PAGES}p 초과" in issue for issue in result["issues"])


def test_missing_font_is_reported_not_skipped(monkeypatch):
    from kv_eval.reporting import export
    from kv_eval.reporting.sections import validate_report

    def no_font():
        raise RuntimeError("no font")
    monkeypatch.setattr(export, "korean_fonts", no_font)
    monkeypatch.delenv("ALLOW_UNCHECKED_PDF", raising=False)
    result = validate_report("## SUMMARY\n근거 부족\n## REFERENCE\n", [])
    assert result["page_check"].startswith("unchecked")
    assert any("PDF 조판 검사 불가" in issue for issue in result["issues"])
    monkeypatch.setenv("ALLOW_UNCHECKED_PDF", "1")
    result = validate_report("## SUMMARY\n근거 부족\n## REFERENCE\n", [])
    assert any("PDF 조판 검사 불가" in w for w in result["warnings"])


def test_export_without_decisions_keeps_existing_run_logs(isolated_outputs, run_graph):
    from kv_eval.reporting.export import korean_fonts
    try:
        korean_fonts()
    except RuntimeError:
        pytest.skip("한글 TrueType 폰트 없음")
    app = _load_app()
    state = run_graph(FakeLLM())
    out = isolated_outputs / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_logs.json").write_text('["keep"]', encoding="utf-8")
    app.finalize({**state, "trace_id": "no-such-trace"}, out)
    assert (out / "run_logs.json").read_text(encoding="utf-8") == '["keep"]'


def test_resume_checks_checkpoint_before_building_runtime(isolated_outputs, monkeypatch):
    app = _load_app()
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "dummy")
    monkeypatch.setenv("CHECKPOINT_PATH", str(isolated_outputs / "ckpt.sqlite"))
    monkeypatch.setattr(app, "build_runtime", lambda *a, **k: pytest.fail("색인을 만들기 전에 실패해야 한다"))
    with pytest.raises(ValueError, match="체크포인트"):
        app.run(resume="no-such-trace")
    with pytest.raises(SystemExit):
        app.parse_args(["--resume", "x", "--query", "q"])


def test_perspective_gap_does_not_exempt_section_structure(run_graph):
    state = run_graph(FakeLLM())
    report = re.sub(r"### 4\.2 시장성\n.*?(?=### 4\.3)", "### 4.2 시장성\n\n", state["report"], flags=re.S)
    gapped = {**state, "report": report, "gaps": ["market: 시장성 재조사 2회 후에도 근거 부족"],
              "perspective_status": {**state["perspective_status"], "market": "excluded"}}
    result = check_coverage(gapped)
    assert result["passed"] is False and result["targets"] == ["market"]
    assert any("판정 표 없음" in i for i in result["issues"])


def test_whole_perspective_gap_exempts_bias_only_when_excluded(run_graph):
    state = run_graph(FakeLLM())
    skewed = [ev for ev in state["evidence"] if not (ev.get("agent") == "market" and ev.get("technology") == "mla")]
    market = {**state["perspectives"]["market"],
              "source_units": {**state["perspectives"]["market"]["source_units"], "mla": []}}
    base = {**state, "evidence": skewed, "perspectives": {**state["perspectives"], "market": market},
            "gaps": ["market: 종합 단계에서 추가 근거 요청"]}
    assert check_bias(base)["passed"] is False  # 관점이 제외되지 않았으면 관점 단위 공백으로 면제하지 않음
    excluded = {**base, "perspective_status": {**state["perspective_status"], "market": "excluded"}}
    assert check_bias(excluded)["passed"] is True
