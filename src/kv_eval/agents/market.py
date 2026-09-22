"""Market evaluation: Tavily ONLY, no local RAG injection."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from hashlib import sha256
from ..schemas import MarketAssessment

from ..tools import retry_queries

# 기준별(M1 시장 규모·성장, M2 상용화·채택, M3 생태계 지지) 검색어. (query, topic)
# M1은 설계 C-2대로 "기술이 속한 시장"을 본다. M3의 공식 문서·릴리스 노트는 뉴스가 아니라 general로 찾는다.
MARKET_QUERIES = {
    "mla": [
        ("LLM inference serving optimization market size growth forecast", "news"),
        ("DeepSeek API inference cost pricing latent attention KV cache", "news"),
        ("DeepSeek V3 R1 available AWS Bedrock NVIDIA NIM Azure", "news"),
        ("vLLM MLA multi-head latent attention backend DeepSeek support", "general"),
        ("SGLang DeepSeek MLA optimization FlashMLA release", "general"),
    ],
    "itme": [
        ("CXL memory market size forecast AI servers", "news"),
        ("SK hynix CXL CMM-DDR5 memory module mass production", "news"),
        ("SK hynix ITME CXL hybrid memory LLM inference", "news"),
        ("CXL memory tiering KV cache LLM inference deployment", "general"),
        ("CXL consortium 3.0 memory pooling standard AI data center adoption", "general"),
    ],
}


def market_node(state: dict, web: Any, llm: Any) -> dict:
    attempt = state["retry_counts"]["market"]
    feedback = state.get("review_feedback", {}).get("market", {})
    results: dict = {}
    evidence: list[dict] = []
    missing: list[str] = []
    for tech, name in (("mla", "DeepSeek-V2 MLA"), ("itme", "ITME CXL hybrid memory")):
        found = []
        queries = MARKET_QUERIES[tech] + [(q, "news") for q in retry_queries(name, feedback)]
        for query, topic in queries:
            found.extend(web.search_market(query, topic))
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
            "Use overall verdict only when supported. Avoid treating absence of adoption announcements as documented objection. "
            "Per design C-2, M1 judges the market the technology BELONGS TO (MLA: LLM inference serving/optimization; ITME: CXL memory). "
            "M2/M3 may also use family-level evidence (MLA: DeepSeek models built on MLA, serving frameworks with an MLA backend; "
            "ITME: SK hynix/CXL memory products) as long as the text labels it '계열 근거' and keeps it distinct from direct adoption of the exact technology. "
            "An explicit MLA kernel/backend in vLLM or SGLang is DIRECT ecosystem support for MLA.\n"
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
