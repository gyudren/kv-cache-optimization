from pathlib import Path
import pytest
from kv_eval.reporting.sections import used_references, validate_report, MANDATORY
from kv_eval.reporting.export import export_report

EVIDENCE = [
    {"source_id": "paper-1", "source_type": "paper", "doc_id": "deepseek_v2", "page": 2,
     "citation_number": 1, "claim": "feature", "excerpt": "original"},
    {"source_id": "web-1", "source_type": "web", "url": "https://example.org/news", "publisher": "Org", "published_at": "", "claim": "adoption"},
]


def report_text() -> str:
    body = "\n\n".join(f"## {title}\n검증 근거 [1, p.2] 및 웹 근거 [W1]에 따라 한계를 기술한다." for title in MANDATORY[:-1])
    refs, _ = used_references(body, EVIDENCE)
    return body + "\n\n## REFERENCE\n" + "\n".join(refs) + "\n"


def test_validate_valid_report():
    assert validate_report(report_text(), EVIDENCE)["passed"]


def test_validate_rejects_invented_reference():
    text = report_text().replace("[1, p.2]", "[1, p.99]", 1)
    assert not validate_report(text, EVIDENCE)["passed"]


def test_report_export_korean_pdf(tmp_path: Path):
    path = export_report(report_text(), str(tmp_path), "test_report")
    assert Path(path["md"]).is_file() and Path(path["pdf"]).is_file()
    from pypdf import PdfReader
    pdf = PdfReader(path["pdf"])
    assert len(pdf.pages) >= 1


def test_summary_page_limit_is_in_gate():
    text = report_text().replace("## SUMMARY\n", "## SUMMARY\n" + "긴 요약 문장입니다. " * 1000)
    outcome = validate_report(text, EVIDENCE)
    assert not outcome["passed"]
    assert any("1/2페이지" in issue for issue in outcome["issues"])
