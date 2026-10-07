"""Technical research: 6 paper questions per selected technology (common 5 + technology-specific mechanism) plus 1 HW baseline.

TRL 판정 근거는 설계 C-1에 따라 두 갈래를 모두 쓴다.
- 논문 원문(RAG): 구현·검증 수준 (TRL 1~6 판단)
- 구현/통합 및 상용화 발표(웹 검색): 제품 출시·상용 서비스 적용 (TRL 7~9 판단)
논문만으로는 "제품 출시·상용 서비스 적용" 근거를 얻을 수 없어 TRL 7~9를 판정할 수 없다.
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

# 기술 공통 질문 + 기술별 작동 원리 질문. MLA는 모델 구조(어텐션) 기술이라 "메모리 계층" 질문을 묻지 않는다
# (공통 질문으로 두면 MLA 답변이 매번 '메모리 계층 설명 없음'을 근거 부족으로 보고해 재조사만 소진했다).
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
# 판정에 꼭 필요한 질문(작동 원리·정량 결과·성숙도 근거). 이 질문의 검색이 근거를 하나도 못 찾으면 필수 결함이다.
# 나머지 질문의 빈칸과 LLM이 적은 세부 미확인 항목(하이퍼파라미터·코드 등)은 선택 항목으로 보고서 한계점에만 남긴다.
REQUIRED_QUESTIONS = {tech: {MECHANISM_QUESTIONS[tech]: "작동 원리", COMMON_QUESTIONS[2]: "정량 성능 결과",
                              COMMON_QUESTIONS[4]: "구현·검증·성숙도 근거"} for tech in MECHANISM_QUESTIONS}
BASELINE_QUESTION = "How do InfiniGen and CXL-PNM differ from ITME in memory expansion mechanism, measured trade-offs, and stated limitations?"

# TRL 7~9(제품 출시·상용 서비스 적용) 판정에 필요한 공개 발표를 찾기 위한 질의
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
    missing = [] if evidence else [f"tech/{tech}: 상용화·통합 웹 검색 결과 없음(TRL 7~9 판정 불가)"]
    return evidence, missing


def technology_node(state: dict, rag: Any, llm: Any, web: Any) -> dict:
    findings: dict = {}
    evidence: list[dict] = []
    missing: list[str] = []    # 필수 결함: Supervisor 재조사 대상
    optional: list[str] = []   # 선택 항목: 보고서 한계점에만 기록
    attempt = attempt_of(state, "tech")
    feedback = state.get("feedback", {}).get("tech", {})
    # 질문끼리 독립이므로 병렬로 검색·답변한다(순서는 아래에서 원래대로 복원).
    # ContextThreadPoolExecutor는 contextvars(LangGraph 실행 설정·LangSmith 부모 run)를 워커 스레드로 복사해
    # 스레드 안의 LLM·웹 호출이 그래프 run 아래 자식 run(같은 trace_id)으로 남게 한다.
    tasks = [(tech, question) for tech in ("mla", "itme") for question in TECH_QUESTIONS[tech]]
    cache = rag_cache.load(state["trace_id"], "tech")
    with ContextThreadPoolExecutor(max_workers=MAX_PARALLEL_QUESTIONS) as pool:
        answers_by_task = list(pool.map(
            lambda item: answer_with_cache(rag, cache, f"[{item[0]}] {item[1]}", item[0], feedback), tasks))
        baseline_future = pool.submit(answer_with_cache, rag, cache, BASELINE_QUESTION, "itme_baseline", feedback)
        web_by_tech = {tech: pool.submit(_trl_web_evidence, web, tech, attempt)
                       for tech in ("mla", "itme")}
        baseline = baseline_future.result()
        web_results = {tech: future.result() for tech, future in web_by_tech.items()}

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
        # 논문(RAG)만으로는 확인할 수 없는 상용화·통합 근거를 웹에서 따로 모은다.
        web_evidence, web_missing = web_results[tech]
        evidence.extend(web_evidence)
        missing.extend(web_missing)
        findings[tech]["trl_web_sources"] = [
            f"{item['source_id']}: {item['publisher']} | {item['excerpt'][:400]}" for item in web_evidence
        ]
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
        for ev in evidence if ev.get("source_type") == "web")
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
        "List in missing only facts you could not confirm; they are reported as limitations. No unsupported claims.\n"
        + "\n".join(f"{k}: {v}" for k, v in findings.items())
        + "\nPaper sources (구현·검증 수준):\n" + paper_refs
        + "\nWeb sources (상용화·통합 발표):\n" + (web_refs or "없음") + rework_note(feedback),
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
