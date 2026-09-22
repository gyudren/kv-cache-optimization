"""Technical research: 6 paper questions per selected technology plus 1 HW baseline.

TRL 판정 근거는 설계 C-1에 따라 두 갈래를 모두 쓴다.
- 논문 원문(RAG): 구현·검증 수준 (TRL 1~6 판단)
- 구현/통합 및 상용화 발표(웹 검색): 제품 출시·상용 서비스 적용 (TRL 7~9 판단)
논문만으로는 "제품 출시·상용 서비스 적용" 근거를 얻을 수 없어 TRL 7~9를 판정할 수 없다.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Any
from ..config import MAX_PARALLEL_QUESTIONS
from ..prompts import prompt_template
from ..schemas import TechnologyAssessment
from ..rag.workflow import answer_with_cache

TECH_QUESTIONS = [
    "What problem and scope does the proposed KV cache approach address?",
    "How does the key-value compression or memory hierarchy mechanism work?",
    "What implementation and experimental setup is documented?",
    "What quantitative memory and throughput/latency outcomes are reported and under what conditions?",
    "What limitations and prerequisites does the paper state?",
    "What evidence exists for implementation, validation, and technology readiness (TRL)?",
]
BASELINE_QUESTION = "How do InfiniGen and CXL-PNM differ from ITME in memory expansion mechanism, measured trade-offs, and stated limitations?"

# TRL 7~9(제품 출시·상용 서비스 적용) 판정에 필요한 공개 발표를 찾기 위한 질의
TRL_WEB_QUERIES = {
    "mla": ["DeepSeek API production service DeepSeek-V2 V3 multi-head latent attention",
            "vLLM MLA attention backend DeepSeek support release",
            "FlashMLA DeepSeek open source MLA decoding kernel"],
    "itme": ["SK hynix ITME CXL hybrid memory inference tiered memory expansion",
             "SK hynix CMM-DDR5 CXL memory module mass production customers",
             "CXL memory module tiered memory LLM inference commercial deployment announcement"],
}
TECH_LABEL = {"mla": "DeepSeek-V2 MLA", "itme": "ITME"}


def _trl_web_evidence(web: Any, tech: str, attempt: int) -> tuple[list[dict], list[str]]:
    """TRL 7~9 판정용 웹 근거를 모은다(설계 C-1의 '구현/통합 및 상용화 발표')."""
    found: list[dict] = []
    for query in TRL_WEB_QUERIES[tech]:
        found.extend(web.search_market(query))
    unique = {item["url"]: item for item in found}.values()
    evidence = [{
        "source_id": f"web:trl:{tech}:{sha256(item['url'].encode()).hexdigest()[:14]}",
        "agent": "tech", "attempt": attempt,
        "claim": f"{TECH_LABEL[tech]}: 구현·통합·상용화 발표(TRL 근거)",
        "excerpt": item["excerpt"], "technology": tech, "source_type": "web",
        "url": item["url"], "title": item["title"],
        "publisher": item["publisher"], "published_at": item["published_at"],
    } for item in unique]
    missing = [] if evidence else [f"{TECH_LABEL[tech]}: 상용화·통합 웹 근거 없음(TRL 7~9 판정 불가)"]
    return evidence, missing


def technology_node(state: dict, rag: Any, llm: Any, web: Any) -> dict:
    findings: dict = {}
    evidence: list[dict] = []
    missing: list[str] = []
    attempt = state["retry_counts"]["tech"]
    feedback = state.get("review_feedback", {}).get("tech", {})
    # 질문끼리 독립이므로 병렬로 검색·답변한다(순서는 아래에서 원래대로 복원).
    tasks = [(tech, question) for tech in ("mla", "itme") for question in TECH_QUESTIONS]
    cache = state.get("tech_result", {}).get("rag_cache", {})
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_QUESTIONS) as pool:
        answers_by_task = list(pool.map(
            lambda item: answer_with_cache(rag, cache, f"[{item[0]}] {item[1]}", item[0], feedback), tasks))
        baseline_future = pool.submit(answer_with_cache, rag, cache, BASELINE_QUESTION, "itme_baseline", feedback)
        web_by_tech = {tech: pool.submit(_trl_web_evidence, web, tech, attempt)
                       for tech in ("mla", "itme")}
        baseline = baseline_future.result()
        web_results = {tech: future.result() for tech, future in web_by_tech.items()}

    for tech in ("mla", "itme"):
        answers = []
        for (task_tech, _), response in zip(tasks, answers_by_task):
            if task_tech != tech:
                continue
            answers.append(response["answer"])
            missing.extend(response["missing"])
            for item in response["evidence"]:
                evidence.append({**item, "agent": "tech", "attempt": attempt})
        findings[tech] = {"answers": answers}
        # 논문(RAG)만으로는 확인할 수 없는 상용화·통합 근거를 웹에서 따로 모은다.
        web_evidence, web_missing = web_results[tech]
        evidence.extend(web_evidence)
        missing.extend(web_missing)
        findings[tech]["trl_web_sources"] = [
            f"{item['source_id']}: {item['publisher']} | {item['excerpt'][:400]}" for item in web_evidence
        ]
    findings["itme_baseline"] = baseline["answer"]
    missing.extend(baseline["missing"])
    evidence.extend({**item, "agent": "tech", "attempt": attempt} for item in baseline["evidence"])
    paper_refs = "\n".join(
        f"[{ev['citation_number']}, p.{ev['page']}] {ev['claim']}: {ev['excerpt'][:300]}"
        for ev in evidence if ev.get("source_type") == "paper")
    web_refs = "\n".join(
        f"{ev['source_id']} ({ev.get('publisher') or '발행 주체 미확인'}, "
        f"{ev.get('published_at') or '게시일 미확인'}): {ev['excerpt'][:300]}"
        for ev in evidence if ev.get("source_type") == "web")
    result = llm.generate_structured(
        prompt_template("technology") + "\n" + "Summarize MLA and ITME technical scope, reported results, limitations and separately assessed TRL from ONLY the supplied excerpts. "
        "Do not confuse product-family deployment with adoption of the exact architecture. "
        "TRL ranges: 1-3 published concept, 4-6 implementation/validation, 7-9 documented operational adoption. "
        "Papers can only support TRL 1-6; assign TRL 7-9 only when a WEB source documents product release or commercial service adoption. "
        "State the maturity of the specific technology separately from the maturity of its general family (for example CXL memory modules in general). "
        "No unsupported claims.\n" + "\n".join(f"{k}: {v}" for k, v in findings.items())
        + "\nPaper sources (구현·검증 수준):\n" + paper_refs
        + "\nWeb sources (상용화·통합 발표):\n" + (web_refs or "없음"),
        TechnologyAssessment,
    )
    missing.extend(result.missing)
    if not all(getattr(result.trl, tech).strip() and getattr(result.trl_basis, tech).strip()
               for tech in ("mla", "itme")):
        missing.append("MLA/ITME의 독립적 TRL 근거 미기재")
    rag_cache = {f"[{tech}] {question}": response for (tech, question), response in zip(tasks, answers_by_task)}
    rag_cache[BASELINE_QUESTION] = baseline
    return {"tech_result": {**result.model_dump(), "details": findings, "rag_cache": rag_cache,
                            "sufficient": result.sufficient and not bool(missing), "missing": list(dict.fromkeys(missing))},
            "evidence": evidence,
            "logs": [{"node": "technology", "attempt": attempt, "result": "complete", "gate": not bool(missing)}]}
