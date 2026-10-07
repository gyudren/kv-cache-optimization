"""기술 조사 Agent: 기술별 논문 질문 6개와 HW 베이스라인 질문 1개로 TRL을 판정한다.

논문(RAG)으로는 TRL 1~6까지만 판단할 수 있어, 7~9의 근거인 제품 출시·상용화 발표는 웹에서 따로 찾는다(설계 C-1).
"""
from __future__ import annotations
from hashlib import sha256
from typing import Any
from ..config import MAX_PARALLEL_QUESTIONS
from langchain_core.runnables.config import ContextThreadPoolExecutor
from ..prompts import prompt_template
from ..tools import rework_note
from ..schemas import TechnologyAssessment
from ..rag import cache as rag_cache
from ..rag.workflow import answer_with_cache
from ..state import attempt_of

# 작동 원리 질문은 기술별로 둔다. MLA는 어텐션 구조 기술이라 메모리 계층을 물으면 근거 부족만 나온다.
COMMON_QUESTIONS = [
    "What problem and scope does the proposed KV cache approach address?",
    "What implementation and experimental setup is documented?",
    "What quantitative memory and throughput/latency outcomes are reported and under what conditions?",
    "What limitations and prerequisites does the paper state?",
    "What evidence exists for implementation, validation, deployment, and technology readiness (TRL)?",
]
MECHANISM_QUESTIONS = {
    "mla": "How does multi-head latent attention compress keys and values into a low-rank latent vector, and how does that change KV cache size per token?",
    "itme": "How does the CXL hybrid memory hierarchy place, tier and prefetch model weights and KV cache across HBM, host DRAM, CXL memory and NVMe?",
}
TECH_QUESTIONS = {tech: [COMMON_QUESTIONS[0], MECHANISM_QUESTIONS[tech], *COMMON_QUESTIONS[1:]] for tech in MECHANISM_QUESTIONS}
# 근거를 하나도 못 찾으면 필수 결함으로 보는 질문. 나머지 질문의 빈칸은 선택 항목으로 한계점에만 남긴다.
REQUIRED_QUESTIONS = {tech: {MECHANISM_QUESTIONS[tech]: "작동 원리", COMMON_QUESTIONS[2]: "정량 성능 결과",
                              COMMON_QUESTIONS[4]: "구현·검증·성숙도 근거"} for tech in MECHANISM_QUESTIONS}
BASELINE_QUESTION = "How do InfiniGen and CXL-PNM differ from ITME in memory expansion mechanism, measured trade-offs, and stated limitations?"

# TRL 7~9 판정용 상용화 발표 검색어
TRL_WEB_QUERIES = {
    "mla": ["DeepSeek-V2 API launch pricing deepseek-chat model release",
            "DeepSeek-V3 multi-head latent attention production inference deployment",
            "vLLM MLA attention backend DeepSeek support release",
            "FlashMLA DeepSeek open source MLA decoding kernel"],
    "itme": ["SK hynix ITME CXL hybrid memory inference tiered memory expansion",
             "SK hynix CMM-DDR5 CXL memory module mass production customers",
             "CXL memory module tiered memory LLM inference commercial deployment announcement"],
}
TECH_LABEL = {"mla": "DeepSeek-V2 MLA", "itme": "ITME"}
# 보고서가 TRL 구간을 쓸 때 인용할 척도 정의 출처(기술 근거가 아님).
TRL_SCALE_QUERY = "NASA technology readiness level TRL 1-9 definitions scale"
TRL_SCALE_SOURCES = 2
TRL_SCALE_CLAIM = "TRL 단계 정의(평가 척도)"


def _trl_scale_evidence(web: Any, attempt: int) -> list[dict]:
    found = web.search_market(TRL_SCALE_QUERY, "general")[:TRL_SCALE_SOURCES]
    return [{
        "source_id": f"web:trl:scale:{sha256(item['url'].encode()).hexdigest()[:14]}",
        "agent": "tech", "attempt": attempt, "claim": TRL_SCALE_CLAIM, "excerpt": item["excerpt"],
        "technology": "general", "source_type": "web", "url": item["url"], "title": item["title"],
        "publisher": item["publisher"], "published_at": item["published_at"],
    } for item in found]


def _trl_web_evidence(web: Any, tech: str, attempt: int) -> tuple[list[dict], list[str]]:
    """TRL 7~9 판정용 구현·통합·상용화 발표를 웹에서 모은다."""
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
    missing = [] if evidence else [f"tech/{tech}: 상용화·통합 웹 검색 결과 없음(TRL 7~9 판정 불가)"]
    return evidence, missing


def technology_node(state: dict, rag: Any, llm: Any, web: Any) -> dict:
    findings: dict = {}
    evidence: list[dict] = []
    # missing은 Supervisor 재조사 대상, optional은 보고서 한계점에만 남는다.
    missing: list[str] = []
    optional: list[str] = []
    attempt = attempt_of(state, "tech")
    feedback = state.get("feedback", {}).get("tech", {})
    # 워커 스레드의 호출도 같은 LangSmith trace에 남도록 ContextThreadPoolExecutor를 쓴다.
    tasks = [(tech, question) for tech in ("mla", "itme") for question in TECH_QUESTIONS[tech]]
    cache = rag_cache.load(state["trace_id"], "tech")
    with ContextThreadPoolExecutor(max_workers=MAX_PARALLEL_QUESTIONS) as pool:
        answers_by_task = list(pool.map(
            lambda item: answer_with_cache(rag, cache, f"[{item[0]}] {item[1]}", item[0], feedback), tasks))
        baseline_future = pool.submit(answer_with_cache, rag, cache, BASELINE_QUESTION, "itme_baseline", feedback)
        web_by_tech = {tech: pool.submit(_trl_web_evidence, web, tech, attempt)
                       for tech in ("mla", "itme")}
        scale_future = pool.submit(_trl_scale_evidence, web, attempt)
        baseline = baseline_future.result()
        web_results = {tech: future.result() for tech, future in web_by_tech.items()}
        scale_evidence = scale_future.result()

    for tech in ("mla", "itme"):
        answers = []
        for (task_tech, question), response in zip(tasks, answers_by_task):
            if task_tech != tech:
                continue
            answers.append(response["answer"])
            optional.extend(response["missing"])
            if question in REQUIRED_QUESTIONS[tech] and not response["evidence"]:
                missing.append(f"tech/{tech}: 필수 질문({REQUIRED_QUESTIONS[tech][question]})의 원문 근거 없음")
            for item in response["evidence"]:
                evidence.append({**item, "agent": "tech", "attempt": attempt})
        findings[tech] = {"answers": answers}
        web_evidence, web_missing = web_results[tech]
        evidence.extend(web_evidence)
        missing.extend(web_missing)
        findings[tech]["trl_web_sources"] = [
            f"{item['source_id']}: {item['publisher']} | {item['excerpt'][:400]}" for item in web_evidence
        ]
    evidence.extend(scale_evidence)
    if not scale_evidence:
        optional.append("TRL 단계 정의 출처를 찾지 못함(보고서는 TRL 구간 대신 근거 수준을 서술)")
    findings["itme_baseline"] = baseline["answer"]
    optional.extend(baseline["missing"])
    if not baseline["evidence"]:
        missing.append("tech/itme: HW 베이스라인(InfiniGen·CXL-PNM) 교차 확인 근거 없음")
    evidence.extend({**item, "agent": "tech", "attempt": attempt} for item in baseline["evidence"])
    paper_refs = "\n".join(
        f"[{ev['citation_number']}, p.{ev['page']}] {ev['claim']}: {ev['excerpt'][:300]}"
        for ev in evidence if ev.get("source_type") == "paper")
    web_refs = "\n".join(
        f"{ev['source_id']} ({ev.get('publisher') or '발행 주체 미확인'}, "
        f"{ev.get('published_at') or '게시일 미확인'}): {ev['excerpt'][:300]}"
        for ev in evidence if ev.get("source_type") == "web" and ev.get("technology") != "general")
    scale_refs = "\n".join(f"{ev['source_id']}: {ev['excerpt'][:300]}" for ev in scale_evidence)
    result = llm.generate_structured(
        prompt_template("technology") + "\n" + "Summarize MLA and ITME technical scope, reported results, limitations and separately assessed TRL from ONLY the supplied excerpts. "
        "TRL ranges: 1-3 published concept, 4-6 implementation/validation, 7-9 documented operational adoption. "
        "Papers alone support TRL 1-6; assign TRL 7-9 when a WEB source documents product release or commercial service adoption "
        "of the technology itself (a first-party paper statement that the system is deployed for service may be cited alongside it). "
        "MLA is the attention architecture of DeepSeek-V2 and its successors (DeepSeek-V3, R1). Production service of these MLA-based "
        "models (DeepSeek API/app, cloud model catalogs) and MLA kernels/backends in serving frameworks are operational adoption of MLA "
        "ITSELF, not of a mere family; do not write that MLA's operation is unconfirmed when such a source or the paper says it is deployed. "
        "For ITME, availability of CXL memory products is family-level evidence only and must not raise ITME's own TRL. "
        "State the maturity of the specific technology separately from the maturity of its general family. "
        "In trl_basis describe the evidence level in words (e.g. 프로토타입·내부 평가 확인, 상용 서비스 운영 확인) and, when something is "
        "not confirmed, narrow it to exactly what is missing (e.g. 독립 재현·외부 고객 채택 미확인) instead of a broad '운영 미확인'. "
        "Cite papers exactly as the supplied [n, p.X] strings (never by document name such as deepseek_v2) and web sources by their source_id. "
        "List in missing only facts you could not confirm; they are reported as limitations. No unsupported claims.\n"
        + "\n".join(f"{k}: {v}" for k, v in findings.items())
        + "\nPaper sources (구현·검증 수준):\n" + paper_refs
        + "\nWeb sources (상용화·통합 발표):\n" + (web_refs or "없음")
        + "\nTRL scale definition sources (척도 정의, 기술 근거 아님):\n" + (scale_refs or "없음") + rework_note(feedback),
        TechnologyAssessment,
    )
    optional.extend(result.missing)
    for tech in ("mla", "itme"):
        if not (getattr(result.trl, tech).strip() and getattr(result.trl_basis, tech).strip()):
            missing.append(f"tech/{tech}: TRL 판정·근거 미기재")
    cached = {f"[{tech}] {question}": response for (tech, question), response in zip(tasks, answers_by_task)}
    cached[BASELINE_QUESTION] = baseline
    missing = list(dict.fromkeys(missing))
    return {"perspectives": {"tech": {**result.model_dump(), "details": findings, "attempt": attempt,
                                      "llm_sufficient": result.sufficient,
                                      "sufficient": not missing, "missing": missing,
                                      "missing_optional": list(dict.fromkeys(optional))}},
            "cache_keys": {"tech": rag_cache.save(state["trace_id"], "tech", cached)},
            "evidence": evidence}
