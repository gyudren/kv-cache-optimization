"""Neutral report using only approved results and verified citation catalog."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from ..schemas import ReportParts
from ..reporting.sections import citeable_evidence, used_references


def render_report(state: dict, llm: Any) -> str:
    sources = citeable_evidence(state["evidence"])
    citation_context = [{"cite": ev["citation"], "claim": ev.get("claim", ""),
                         "excerpt": ev.get("excerpt", "")[:550], "url": ev.get("url", "")}
                        for ev in sources]
    sections = llm.generate_structured(
        prompt_template("report") + "\n" + "Write Korean evidence-based multi-perspective evaluation report of DeepSeek-V2 MLA and ITME for datacenter/cloud long-context LLM serving. "
        "Provide separate fields for SUMMARY, background, selection, technology overview, perspectives (TRL, market, stakeholder, D1-D7 for each), "
        "synthesis (agreements/conflicts/relations), implications, limitations. "
        "SUMMARY concise for <= half A4 page. No ranking/endorsement. No invented deployment or quantitative results. "
        "The limitations section MUST state explicitly that every TRL judgement is an estimate based on public information only "
        "(papers, patents, commercial announcements) and that there is a lag between publication and actual adoption, "
        "and MUST list the confirmation-bias countermeasures actually taken (HW baseline papers used to cross-check ITME, "
        "identical reporting format for both technologies, synthesis agent restricted to already-verified evidence). "
        "Use the EXACT citation strings from the provided source catalog in the text for every supported fact. "
        "Never cite non-catalog ID. If missing, mark 근거 부족 and show evidence gaps. "
        "Avoid changing design category labels.\n"
        + repr({k: state.get(k, {}) for k in ("tech_result", "market_result", "stakeholder_result", "domain_result", "synthesis_result")})
        + "\nVerified source catalog:\n" + repr(citation_context)
        + "\nRevision feedback (address every issue):\n" + repr(state.get("review_feedback", {}).get("report", {}))
        + "\nPrevious draft to revise:\n" + state.get("report_draft", ""), ReportParts,
    )
    parts = sections.model_dump()
    report = "\n\n".join([
        f"## SUMMARY\n{parts['summary']}",
        f"## 1. 분석 배경\n{parts['background']}",
        f"## 2. 기술 선정\n{parts['selection']}",
        f"## 3. 기술 개요\n{parts['technology_overview']}",
        f"## 4. 관점별 평가\n{parts['perspectives']}",
        f"## 5. 종합 의견\n{parts['synthesis']}",
        f"## 6. 시사점\n{parts['implications']}",
        f"## 7. 한계점\n{parts['limitations']}",
    ])
    refs, _ = used_references(report, state["evidence"])
    report += "\n\n## REFERENCE\n" + ("\n".join(refs) if refs else "근거 부족: 실제 사용한 검증 가능 참고자료 없음") + "\n"
    return report


def report_node(state: dict, llm: Any) -> dict:
    report = render_report(state, llm)
    return {"report_draft": report,
            "logs": [{"node": "report", "attempt": state["retry_counts"]["report"], "result": "complete"}]}
