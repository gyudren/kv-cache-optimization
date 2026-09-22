"""E: eight mandatory sections, provenance and citation validation."""
from __future__ import annotations
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from ..state import deduplicate_evidence


@lru_cache(maxsize=1)
def bibliography() -> dict[str, dict]:
    """manifest.json의 서지 정보를 doc_id로 찾을 수 있게 읽어 둔다.

    확인되지 않은 항목(학회/권호 등)은 추정하지 않고 '미확인'으로 표기한다.
    """
    path = Path(os.getenv("MANIFEST_PATH", "data/manifest.json"))
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {doc["doc_id"]: doc for doc in manifest.get("documents", [])}


def format_paper_reference(number: int, doc_id: str, pages: list[int]) -> str:
    """가이드 REFERENCE 형식(논문): 저자(YYYY). 논문제목. 학술지/학회명, 권(호), 페이지.

    확인되지 않은 항목은 추정하지 않고 '미확인'으로 적는다. 페이지에는 보고서가 실제로
    인용한 원문 페이지만 기재한다(활용한 자료만 기재한다는 규칙에 맞춘다).
    """
    meta = bibliography().get(doc_id, {})
    authors = meta.get("authors") or "저자 미확인"
    year = meta.get("year") or "연도 미확인"
    title = meta.get("full_title") or meta.get("title") or doc_id
    if meta.get("venue"):
        venue = meta["venue"]
    elif meta.get("arxiv_id"):
        venue = f"arXiv preprint arXiv:{meta['arxiv_id']}"
    else:
        venue = "학술지/학회명 미확인"
    cited = ", ".join(f"p.{page}" for page in sorted(set(pages)))
    return f"[{number}] {authors} ({year}). {title}. {venue}, {cited}."

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
    for ev in deduplicate_evidence(evidence):
        if ev.get("source_type") == "paper" and ev.get("citation_number") and ev.get("page"):
            ret.append({**ev, "citation": f"[{ev['citation_number']}, p.{ev['page']}]"})
    for i, ev in catalog["web"].items():
        ret.append({**ev, "citation": f"[W{i}]"})
    return ret


def used_references(report: str, evidence: list[dict]) -> tuple[list[str], list[str]]:
    catalog = citation_catalog(evidence)
    issues = []
    refs = []
    # Do not count references themselves as proof of an in-text citation.
    body = report.split("\n## REFERENCE", 1)[0]
    # 같은 논문의 여러 페이지를 인용하면 REFERENCE에는 한 항목으로 모아 쓴다(설계 E).
    paper_pages: dict[tuple[int, str], list[int]] = {}
    for n_str, p_str in dict.fromkeys(PAPER_CITATION.findall(body)):
        key = (int(n_str), int(p_str))
        ev = catalog["paper"].get(key)
        if ev is None:
            issues.append(f"인용 [{n_str}, p.{p_str}]에 대응하는 검증된 원문 근거 없음")
        else:
            paper_pages.setdefault((int(n_str), ev["doc_id"]), []).append(int(p_str))
    for (number, doc_id), pages in sorted(paper_pages.items()):
        refs.append(format_paper_reference(number, doc_id, pages))
    for number in dict.fromkeys(int(x) for x in WEB_CITATION.findall(body)):
        ev = catalog["web"].get(number)
        if ev is None:
            issues.append(f"웹 인용 [W{number}]의 URL 근거 없음")
        else:
            # 가이드 REFERENCE 형식(기타/웹): 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL
            publisher = ev.get("publisher") or "발행 주체 미확인"
            date = ev.get("published_at") or "게시일 미확인"
            title = ev.get("title") or "제목 미확인"
            refs.append(f"[W{number}] {publisher} ({date}). {title}. {publisher}, {ev['url']}")
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
