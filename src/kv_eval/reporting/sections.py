"""E: eight mandatory sections, provenance and citation validation."""
from __future__ import annotations
import re
from ..state import deduplicate_evidence

MANDATORY = ["SUMMARY", "1. 분석 배경", "2. 기술 선정", "3. 기술 개요", "4. 관점별 평가", "5. 종합 의견", "6. 시사점", "7. 한계점", "REFERENCE"]
PAPER_CITATION = re.compile(r"\[(\d+),\s*p\.(\d+)\]")
WEB_CITATION = re.compile(r"\[W(\d+)\]")


def citation_catalog(evidence: list[dict]) -> dict:
    """Web sources get stable report-local W indices; paper numbering from manifest."""
    unique = deduplicate_evidence(evidence)
    paper = {(int(ev["citation_number"]), int(ev["page"])): ev for ev in unique
             if ev.get("source_type") == "paper" and ev.get("citation_number") and ev.get("page")}
    urls = sorted({ev["url"] for ev in unique if ev.get("source_type") == "web" and ev.get("url")})
    web = {i + 1: next(ev for ev in unique if ev.get("url") == url) for i, url in enumerate(urls)}
    return {"paper": paper, "web": web}


def citeable_evidence(evidence: list[dict]) -> list[dict]:
    catalog = citation_catalog(evidence)
    ret = []
    for (number, page), ev in catalog["paper"].items():
        ret.append({**ev, "citation": f"[{number}, p.{page}]"})
    for i, ev in catalog["web"].items():
        ret.append({**ev, "citation": f"[W{i}]"})
    return ret


def used_references(report: str, evidence: list[dict]) -> tuple[list[str], list[str]]:
    catalog = citation_catalog(evidence)
    issues = []
    refs = []
    # Do not count references themselves as proof of an in-text citation.
    body = report.split("\n## REFERENCE", 1)[0]
    for n_str, p_str in dict.fromkeys(PAPER_CITATION.findall(body)):
        key = (int(n_str), int(p_str))
        ev = catalog["paper"].get(key)
        if ev is None:
            issues.append(f"인용 [{n_str}, p.{p_str}]에 대응하는 검증된 원문 근거 없음")
        else:
            refs.append(f"[{n_str}] {ev['doc_id']}, PDF p.{p_str} (원문 출처; 서지 정보는 별도 확인 필요)")
    for number in dict.fromkeys(int(x) for x in WEB_CITATION.findall(body)):
        ev = catalog["web"].get(number)
        if ev is None:
            issues.append(f"웹 인용 [W{number}]의 URL 근거 없음")
        else:
            publisher = ev.get("publisher") or "발행 주체 미확인"
            date = ev.get("published_at") or "게시일 미확인"
            refs.append(f"[W{number}] {publisher} ({date}). {ev['url']}")
    return refs, issues


def validate_report(report: str, evidence: list[dict]) -> dict:
    issues = []
    positions = []
    for chapter in MANDATORY:
        marker = f"## {chapter}"
        count = report.count(marker + "\n")
        if count != 1:
            issues.append(f"필수 목차 중복/누락: {chapter} ({count}회)")
        else:
            positions.append(report.index(marker))
    if positions != sorted(positions):
        issues.append("목차 순서가 설계 E와 불일치")
    if "## SUMMARY\n" in report and "\n## 1. 분석 배경" in report:
        from .export import summary_fits_half_page
        summary = report.split("## SUMMARY\n", 1)[1].split("\n## 1. 분석 배경", 1)[0].strip()
        if not summary_fits_half_page(summary):
            issues.append("SUMMARY가 PDF 렌더링 기준 1/2페이지 초과")
    refs, citation_issues = used_references(report, evidence)
    issues.extend(citation_issues)
    if not PAPER_CITATION.search(report) and not WEB_CITATION.search(report):
        issues.append("본문에 검증 가능한 인용 없음")
    reference_text = report.split("\n## REFERENCE\n", 1)[-1] if "\n## REFERENCE\n" in report else ""
    if refs and any(ref not in reference_text for ref in refs):
        issues.append("실제 사용한 근거가 REFERENCE에 누락")
    if not refs and "근거 부족" not in report:
        issues.append("검증 가능한 출처 및 근거 부족 표기 모두 없음")
    return {"passed": not issues, "issues": issues, "used_references": refs}
