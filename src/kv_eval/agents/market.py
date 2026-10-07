"""시장성 평가: 로컬 RAG 없이 웹 검색 결과만 근거로 쓴다."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from hashlib import sha256
from ..schemas import MarketAssessment

from ..tools import retry_queries, rework_note
from ..state import attempt_of

# 기준별(M1 시장 규모·성장, M2 상용화·채택, M3 생태계 지지) 검색어와 topic
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


# 이 중 하나라도 판정했으면 관점 결과가 성립한다. 나머지는 표에 '근거 부족'으로 남는다.
CRITERION_VERDICTS = ("market_size_growth_verdict", "adoption_verdict", "ecosystem_verdict")


def market_node(state: dict, web: Any, llm: Any) -> dict:
    attempt = attempt_of(state, "market")
    feedback = state.get("feedback", {}).get("market", {})
    results: dict = {}
    evidence: list[dict] = []
    # missing은 Supervisor 재조사 대상, optional은 보고서 한계점에만 남는다.
    missing: list[str] = []
    optional: list[str] = []
    for tech, name in (("mla", "DeepSeek-V2 MLA"), ("itme", "ITME CXL hybrid memory")):
        found = []
        queries = MARKET_QUERIES[tech] + [(q, "news") for q in retry_queries(name, feedback, tech=tech)]
        for query, topic in queries:
            found.extend(web.search_market(query, topic))
        # URL로 중복을 없애고, source_id는 URL 해시라 재실행해도 같다.
        sources = {entry["url"]: entry for entry in found}
        raw = [{**r, "source_id": f"web:{tech}:{sha256(r['url'].encode()).hexdigest()[:14]}"} for r in sources.values()]
        valid_ids = {r["source_id"] for r in raw}
        if not raw:
            missing.append(f"market/{tech}: 시장·채택·생태계 웹 검색 결과 없음")
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
            "An explicit MLA kernel/backend in vLLM or SGLang is DIRECT ecosystem support for MLA. "
            "MLA is the attention architecture of DeepSeek-V2/V3/R1, so DeepSeek's own API/app serving these models and cloud catalogs "
            "offering them are DIRECT commercial adoption evidence for MLA (M2), not merely family-level evidence. "
            "Put only facts you could not confirm in missing; they are reported as limitations.\n"
            + "\n".join(f"{r['source_id']}: {r['title']} | {r['url']} | {r['excerpt'][:1400]}" for r in raw) + rework_note(feedback),
            MarketAssessment,
        )
        cited = [r for r in raw if r["source_id"] in set(assessment.cited_ids) & valid_ids]
        issues = []
        if not cited:
            issues.append(f"market/{tech}: 검증 가능한 출처 인용 없음")
        if not any(getattr(assessment, field) for field in CRITERION_VERDICTS):
            issues.append(f"market/{tech}: M1~M3 어느 기준도 판정하지 못함")
        missing.extend(issues)
        optional.extend(assessment.missing)
        results[tech] = {**assessment.model_dump(), "llm_sufficient": assessment.sufficient,
                         "sufficient": not issues, "missing": issues}
        evidence.extend({"source_id": r["source_id"], "agent": "market", "attempt": attempt,
                         "claim": f"{name}: 시장·채택·생태계", "excerpt": r["excerpt"],
                         "technology": tech, "source_type": "web", "url": r["url"], "title": r["title"],
                         "publisher": r["publisher"], "published_at": r["published_at"]} for r in cited)
    return {"perspectives": {"market": {"technologies": results, "attempt": attempt,
                                       "llm_sufficient": all(r.get("llm_sufficient", False) for r in results.values()),
                                       "sufficient": all(r.get("sufficient", False) for r in results.values()),
                                       "missing": list(dict.fromkeys(missing)),
                                       "missing_optional": list(dict.fromkeys(optional))}},
            "evidence": evidence}
