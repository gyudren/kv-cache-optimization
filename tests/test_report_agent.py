from pathlib import Path
from kv_eval.agents.report import render_report
from kv_eval.reporting.export import export_report, markdown_flowables
from kv_eval.reporting.sections import validate_report
from kv_eval.schemas import ReportParts
from kv_eval.state import initial_state

EVIDENCE = [
    {"source_id": "chunk-mla-7", "source_type": "paper", "doc_id": "deepseek_v2_mla", "page": 7,
     "citation_number": 1, "technology": "mla", "agent": "domain", "claim": "KV cache", "excerpt": "93.3%"},
    {"source_id": "chunk-itme-3", "source_type": "paper", "doc_id": "itme", "page": 3,
     "citation_number": 2, "technology": "itme", "agent": "domain", "claim": "CXL", "excerpt": "tiered"},
    {"source_id": "web:mla:abc", "source_type": "web", "url": "https://example.org/mla", "technology": "mla",
     "agent": "market", "publisher": "example.org", "published_at": "2026-01-02", "title": "MLA", "claim": "m"},
]


class ReportLLM:
    def __init__(self):
        self.prompts = []

    def generate_structured(self, prompt, schema):
        assert schema is ReportParts
        self.prompts.append(prompt)
        text = "근거 문장 [1, p.7] [2, p.3] [W1]."
        return ReportParts(summary="- 요약 [1, p.7]", background=text, selection=text,
                           technology_overview="| 항목 | MLA | ITME |\n|---|---|---|\n| 계층 | 모델 | 메모리 |",
                           perspective_trl=text, perspective_market=text, perspective_stakeholder=text,
                           perspective_domain=text, synthesis=text, implications=text, limitations=text)


def state():
    s = initial_state("q")
    s["evidence"] = EVIDENCE
    s["tech_result"] = {"trl": {"mla": "TRL 9", "itme": "TRL 4"}, "trl_basis": {"mla": "상용", "itme": "논문"}}
    s["market_result"] = {"technologies": {"mla": {"adoption_verdict": "긍정", "cited_ids": ["web:mla:abc"]},
                                           "itme": {}}}
    s["domain_result"] = {"items": [
        {"dimension": "D1 워크로드 수용 능력", "technology": "mla", "verdict": "적합", "cited_ids": ["chunk-mla-7"]},
        {"dimension": "D1 워크로드 수용 능력", "technology": "itme", "verdict": "조건부", "cited_ids": ["chunk-itme-3"]},
    ]}
    return s


def test_report_tables_follow_agent_verdicts_and_validate():
    report = render_report(state(), ReportLLM())
    assert "| D1 워크로드 수용 능력 | 적합 | [1, p.7] | 조건부 | [2, p.3] |" in report
    assert "| 상용화·채택 현황 | 긍정 | 근거 부족 |" in report
    assert "| DeepSeek-V2 MLA | TRL 9 |" in report
    assert "증거 균형표" in report and "2,560 GiB" in report and "TurboQuant" in report
    outcome = validate_report(report, EVIDENCE)
    assert outcome["passed"], outcome["issues"]


def test_report_retry_receives_gate_feedback():
    s = state()
    s["review_feedback"] = {"report": {"issues": ["SUMMARY가 PDF 렌더링 기준 1/2페이지 초과"]}}
    llm = ReportLLM()
    render_report(s, llm)
    assert "SUMMARY가 PDF 렌더링 기준 1/2페이지 초과" in llm.prompts[0]


def test_markdown_tables_render_as_pdf_tables(tmp_path: Path):
    from reportlab.platypus import Table
    flowables = markdown_flowables("#### 표\n| a | b |\n|---|---|\n| **1** | 2 |\n\n- 항목")
    assert any(isinstance(f, Table) for f in flowables)
    paths = export_report(render_report(state(), ReportLLM()), str(tmp_path), "t")
    assert Path(paths["pdf"]).stat().st_size > 0
