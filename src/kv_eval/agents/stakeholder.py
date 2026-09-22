"""Stakeholder evidence with identifiable attributed voices (Tavily web only)."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from hashlib import sha256
from ..schemas import StakeholderAssessment

DIMENSIONS = ("competitor reactions", "developer enterprise adopter experience", "investor industry analyst perspective")


def stakeholder_node(state: dict, web: Any, llm: Any) -> dict:
    attempt = state["retry_counts"]["stakeholder"]
    feedback = state.get("review_feedback", {}).get("stakeholder", {})
    results, evidence, missing = {}, [], []
    for tech, name in (("mla", "DeepSeek-V2 MLA"), ("itme", "ITME CXL hybrid memory")):
        found = []
        for dimension in DIMENSIONS:
            found.extend(web.search_stakeholder(f"{name} {dimension} {feedback.get('rewritten_queries', [])}"))
        raw = [{**r, "source_id": f"web:stakeholder:{tech}:{sha256(r['url'].encode()).hexdigest()[:14]}"}
               for r in {x["url"]: x for x in found}.values()]
        if not raw:
            issue = f"{name}: 이해관계자 웹 자료 없음"
            missing.append(issue)
            results[tech] = {"sufficient": False, "missing": [issue], "summary": "근거 부족"}
            continue
        assessment = llm.generate_structured(
            prompt_template("stakeholder") + "\n" + "Each source line starts with a speaker hint derived from its domain (언론/개발자 커뮤니티/기업 공식 발표/투자·애널리스트/기타); "
            "use it as a candidate only and confirm the actual speaker from the text. "
            "Evaluate ONLY explicitly ATTRIBUTED stakeholder statements by competitors, developers/adopters, investors. "
            "A competitor developing an alternative is not itself proof of a negative judgment. "
            "Record who said what; never attribute anonymous text to an imagined speaker. "
            "Set separate verdicts for competitors, developers/adopters and investors only with explicit support (otherwise null). "
            "Use only provided source IDs in cited_ids; absence of information means missing.\n"
            + "\n".join(f"{r['source_id']}: [{r.get('speaker', '기타')}] {r['title']} | {r['url']} | {r['excerpt'][:1400]}" for r in raw),
            StakeholderAssessment,
        )
        cited = [r for r in raw if r["source_id"] in set(assessment.cited_ids)]
        issues = list(assessment.missing)
        if not cited:
            issues.append(f"{name}: 이해관계자 의견의 검증 가능한 인용 없음")
        missing.extend(issues)
        results[tech] = {**assessment.model_dump(), "sufficient": assessment.sufficient and bool(cited) and not bool(issues), "missing": issues}
        evidence.extend({"source_id": r["source_id"], "agent": "stakeholder", "attempt": attempt,
                         "claim": f"{name}: 이해관계자 평가", "excerpt": r["excerpt"],
                         "technology": tech, "source_type": "web", "url": r["url"],
                         "publisher": r["publisher"], "published_at": r["published_at"],
                         "speaker": r.get("speaker", "기타")} for r in cited)
    return {"stakeholder_result": {"technologies": results, "sufficient": all(r.get("sufficient", False) for r in results.values()),
                                   "missing": list(dict.fromkeys(missing))}, "evidence": evidence,
            "logs": [{"node": "stakeholder", "attempt": attempt, "result": "complete", "gate": not bool(missing)}]}
