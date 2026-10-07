"""이해관계자 평가: 발언 주체가 확인되는 웹 자료(Tavily)만 근거로 쓴다."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from hashlib import sha256
from ..schemas import StakeholderAssessment

from ..tools import retry_queries, rework_note
from ..state import attempt_of
from ..tools.web_search import COMMUNITY_SPEAKER

# 평가 대상(경쟁 진영, 도입 기업·개발자, 투자 업계)별 검색어
STAKEHOLDER_QUERIES = {
    "mla": [
        "DeepSeek multi-head latent attention competitors response OpenAI Meta Google",
        "MLA vs GQA latent attention KV cache engineers analysis",
        "vLLM SGLang developers DeepSeek MLA kernel performance experience",
        "DeepSeek efficiency analysts investors reaction Nvidia inference cost",
    ],
    "itme": [
        "Samsung Micron CXL memory expansion AI inference competition SK hynix",
        "CXL memory expansion hyperscaler cloud adoption LLM inference latency concerns",
        "SK hynix CXL memory analyst outlook AI demand",
        "CXL vs HBM AI memory industry view",
    ],
}


# 이 중 하나라도 판정했으면 관점 결과가 성립한다.
CRITERION_VERDICTS = ("competitors_verdict", "developers_adopters_verdict", "investors_verdict")


def stakeholder_node(state: dict, web: Any, llm: Any) -> dict:
    attempt = attempt_of(state, "stakeholder")
    feedback = state.get("feedback", {}).get("stakeholder", {})
    # missing은 Supervisor 재조사 대상, optional은 보고서 한계점에만 남는다.
    results, evidence, missing, optional = {}, [], [], []
    for tech, name in (("mla", "DeepSeek-V2 MLA"), ("itme", "ITME CXL hybrid memory")):
        found = []
        for query in STAKEHOLDER_QUERIES[tech] + retry_queries(name, feedback, tech=tech):
            found.extend(web.search_stakeholder(query))
        raw = [{**r, "source_id": f"web:stakeholder:{tech}:{sha256(r['url'].encode()).hexdigest()[:14]}"}
               for r in {x["url"]: x for x in found}.values()]
        if not raw:
            issue = f"stakeholder/{tech}: 이해관계자 웹 검색 결과 없음"
            missing.append(issue)
            results[tech] = {"sufficient": False, "missing": [issue], "summary": "근거 부족"}
            continue
        assessment = llm.generate_structured(
            prompt_template("stakeholder") + "\n" + "Each source line starts with a speaker hint derived from its domain "
            f"(언론/{COMMUNITY_SPEAKER}/개발자 공식 저장소/기업·기술 블로그/기업 공식 발표/투자·애널리스트/기타); "
            "use it as a candidate only and confirm the actual speaker from the text. "
            f"Sources marked {COMMUNITY_SPEAKER} (forums, Hacker News, Reddit, personal blogs) are individual opinions: they may support "
            "ONLY the developers/adopters verdict, labelled '개발자 커뮤니티 의견', never the competitors or investors verdict. "
            "Evaluate ONLY explicitly ATTRIBUTED stakeholder statements by competitors, developers/adopters, investors. "
            "A competitor developing an alternative is not itself proof of a negative judgment. "
            "Record who said what; never attribute anonymous text to an imagined speaker. "
            "Set separate verdicts for competitors, developers/adopters and investors only with explicit support (otherwise null). "
            "Use only provided source IDs in cited_ids; absence of information means missing. "
            "Statements about the technology family (MLA: DeepSeek's MLA-based models and MLA serving kernels; ITME: SK hynix/CXL memory expansion) "
            "may support a verdict when the speaker is identified and the text labels them '계열·간접 반응', distinct from reactions to the exact technology.\n"
            + "\n".join(f"{r['source_id']}: [{r.get('speaker', '기타')}] {r['title']} | {r['url']} | {r['excerpt'][:1400]}" for r in raw) + rework_note(feedback),
            StakeholderAssessment,
        )
        cited = [r for r in raw if r["source_id"] in set(assessment.cited_ids)]
        if cited and all(r.get("speaker") == COMMUNITY_SPEAKER for r in cited):
            # 개인·커뮤니티 글만으로는 경쟁 진영·투자 업계 반응을 판정하지 않는다.
            held = [f for f in ("competitors_verdict", "investors_verdict") if getattr(assessment, f)]
            if held:
                assessment = assessment.model_copy(update={f: None for f in held})
                optional.append(f"stakeholder/{tech}: 경쟁 진영·투자 업계 반응은 개인·커뮤니티 글뿐이라 판정 보류")
        issues = []
        if not cited:
            issues.append(f"stakeholder/{tech}: 이해관계자 의견의 검증 가능한 인용 없음")
        if not any(getattr(assessment, field) for field in CRITERION_VERDICTS):
            issues.append(f"stakeholder/{tech}: S1~S3 어느 이해관계자 유형도 판정하지 못함")
        missing.extend(issues)
        optional.extend(assessment.missing)
        results[tech] = {**assessment.model_dump(), "llm_sufficient": assessment.sufficient,
                         "sufficient": not issues, "missing": issues}
        evidence.extend({"source_id": r["source_id"], "agent": "stakeholder", "attempt": attempt,
                         "claim": f"{name}: 이해관계자 평가", "excerpt": r["excerpt"],
                         "technology": tech, "source_type": "web", "url": r["url"], "title": r["title"],
                         "publisher": r["publisher"], "published_at": r["published_at"],
                         "speaker": r.get("speaker", "기타")} for r in cited)
    return {"perspectives": {"stakeholder": {"technologies": results, "attempt": attempt,
                                       "llm_sufficient": all(r.get("llm_sufficient", False) for r in results.values()),
                                       "sufficient": all(r.get("sufficient", False) for r in results.values()),
                                       "missing": list(dict.fromkeys(missing)),
                                       "missing_optional": list(dict.fromkeys(optional))}},
            "evidence": evidence}
