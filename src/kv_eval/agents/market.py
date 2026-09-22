"""Market evaluation: Tavily ONLY, no local RAG injection."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from hashlib import sha256
from ..schemas import MarketAssessment

MARKET_DIMENSIONS = ("market size and growth", "commercial adoption", "serving ecosystem and standardization")


def market_node(state: dict, web: Any, llm: Any) -> dict:
    attempt = state["retry_counts"]["market"]
    feedback = state.get("review_feedback", {}).get("market", {})
    results: dict = {}
    evidence: list[dict] = []
    missing: list[str] = []
    for tech, name in (("mla", "DeepSeek-V2 MLA"), ("itme", "ITME CXL hybrid memory")):
        found = []
        for dimension in MARKET_DIMENSIONS:
            found.extend(web.search_market(f"{name} {dimension} {feedback.get('rewritten_queries', [])}"))
        # Distinct URL, stable citation IDs; web excerpt is the only supporting text.
        sources = {entry["url"]: entry for entry in found}
        raw = [{**r, "source_id": f"web:{tech}:{sha256(r['url'].encode()).hexdigest()[:14]}"} for r in sources.values()]
        valid_ids = {r["source_id"] for r in raw}
        if not raw:
            missing.append(f"{name}: 시장·채택·생태계 웹 근거 없음")
            results[tech] = {"sufficient": False, "missing": [missing[-1]], "summary": "근거 부족"}
            continue
        assessment = llm.generate_structured(
            prompt_template("market") + "\n" + "Evaluate market size/growth, commercial adoption and serving ecosystem from WEB SOURCES ONLY. "
            "Use only source IDs actually supplied in cited_ids; if source is insufficient set sufficient=False and missing. "
            "Record a separate 긍정/우려/혼재 verdict for each of market_size_growth, adoption and ecosystem only when supported, else null. "
            "Use overall verdict only when supported. Avoid treating absence of adoption announcements as documented objection.\n"
            + "\n".join(f"{r['source_id']}: {r['title']} | {r['url']} | {r['excerpt'][:1400]}" for r in raw),
            MarketAssessment,
        )
        cited = [r for r in raw if r["source_id"] in set(assessment.cited_ids) & valid_ids]
        issues = list(assessment.missing)
        if not cited:
            issues.append(f"{name}: 검증 가능한 출처 인용 없음")
        missing.extend(issues)
        results[tech] = {**assessment.model_dump(), "sufficient": assessment.sufficient and bool(cited) and not bool(issues), "missing": issues}
        evidence.extend({"source_id": r["source_id"], "agent": "market", "attempt": attempt,
                         "claim": f"{name}: 시장·채택·생태계", "excerpt": r["excerpt"],
                         "technology": tech, "source_type": "web", "url": r["url"], "title": r["title"],
                         "publisher": r["publisher"], "published_at": r["published_at"]} for r in cited)
    return {"market_result": {"technologies": results, "sufficient": all(r.get("sufficient", False) for r in results.values()),
                              "missing": list(dict.fromkeys(missing))}, "evidence": evidence,
            "logs": [{"node": "market", "attempt": attempt, "result": "complete", "gate": not bool(missing)}]}
