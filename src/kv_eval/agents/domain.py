"""도메인 적용성 평가: 공통 D1~D7 항목을 기술별 원문 논문만 근거로 판정한다."""
from __future__ import annotations
import re
from typing import Any
from ..config import MAX_PARALLEL_QUESTIONS
from langchain_core.runnables.config import ContextThreadPoolExecutor
from ..prompts import prompt_template
from ..tools import rework_note
from ..schemas import DomainAssessment
from ..rag import cache as rag_cache
from ..rag.workflow import answer_with_cache
from ..state import attempt_of

DIMENSIONS = [
    ("D1", "워크로드 수용 능력", "context and concurrent requests given constrained GPU memory"),
    ("D2", "서비스 성능", "generation throughput, prompt throughput, TTFT and latency under memory pressure"),
    ("D3", "메모리 자원 효율", "KV bytes/token and HBM, host DRAM, CXL/NVMe consumption"),
    ("D4", "품질·정보 보존", "reported output quality and retention of long-context information"),
    ("D5", "운영 안정성", "multi-turn concurrency, cache misses, contention and performance variation"),
    ("D6", "도입·확장 용이성", "model architecture/weights vs CXL, vLLM, RDMA or NIC requirements"),
    ("D7", "비용 효율성", "reported infrastructure cost or resources per equivalent workload"),
]


CANONICAL = {code: f"{code} {title}" for code, title, _ in DIMENSIONS}


def normalize_items(items: list[dict], evidence: list[dict]) -> list[dict]:
    """LLM이 쓴 평가항목명은 D-코드 정식 이름으로, "[1, p.7]" 형식 인용은 source_id로 맞춘다.

    맞는 source_id가 없는 인용은 그대로 남겨 검증에서 걸리게 한다.
    """
    by_citation: dict[str, list[str]] = {}
    for ev in evidence:
        by_citation.setdefault(f"[{ev['citation_number']}, p.{ev['page']}]", []).append(ev["source_id"])
    valid = {ev["source_id"] for ev in evidence}
    out = []
    for item in items:
        match = re.search(r"D\s*([1-7])", item["dimension"])
        dimension = CANONICAL[f"D{match.group(1)}"] if match else item["dimension"]
        cited: list[str] = []
        for cid in item.get("cited_ids", []):
            key = re.sub(r"\s+", " ", cid.strip()).replace("[ ", "[").replace(" ]", "]")
            if cid in valid:
                cited.append(cid)
            elif key in by_citation:
                cited.extend(by_citation[key])
            else:
                cited.append(cid)
        out.append({**item, "dimension": dimension, "cited_ids": list(dict.fromkeys(cited))})
    return out


SINGLE_DOC_NOTE = " (근거가 원문 1편의 저자 보고에 한정되어 '조건부'로 둠)"


def downgrade_single_document(items: list[dict], evidence: list[dict]) -> list[dict]:
    """'적합' 판정의 근거 문서가 1편뿐이면 '조건부'로 낮춘다.

    같은 논문의 여러 페이지는 독립 출처가 아니라서 저자 보고 하나로 '적합'을 확정하지 않는다.
    """
    doc_of = {ev["source_id"]: ev.get("doc_id") for ev in evidence}
    out = []
    for item in items:
        docs = {doc_of[cid] for cid in item.get("cited_ids", []) if doc_of.get(cid)}
        if item.get("verdict") == "적합" and len(docs) <= 1:
            item = {**item, "verdict": "조건부", "explanation": item.get("explanation", "") + SINGLE_DOC_NOTE}
        out.append(item)
    return out


def domain_node(state: dict, rag: Any, llm: Any) -> dict:
    attempt = attempt_of(state, "domain")
    feedback = state.get("feedback", {}).get("domain", {})
    research = []
    evidence = []
    # missing은 Supervisor 재조사 대상, optional은 보고서 한계점에만 남는다.
    missing = []
    optional = []
    cache = rag_cache.load(state["trace_id"], "domain")
    tasks = [(tech, code, title, question)
             for tech in ("mla", "itme") for code, title, question in DIMENSIONS]
    # 워커 스레드의 호출도 같은 LangSmith trace에 남도록 ContextThreadPoolExecutor를 쓴다.
    with ContextThreadPoolExecutor(max_workers=MAX_PARALLEL_QUESTIONS) as pool:
        answers = list(pool.map(
            lambda t: answer_with_cache(rag, cache, f"{t[0]} {t[1]} {t[2]}: {t[3]}", t[0], feedback), tasks))
    for (tech, code, title, question), answer in zip(tasks, answers):
        research.append({"technology": tech, "dimension": f"{code} {title}", "question": question,
                         "answer": answer["answer"], "sufficient": answer["sufficient"]})
        optional.extend(answer["missing"])
        evidence.extend({**item, "agent": "domain", "attempt": attempt} for item in answer["evidence"])
    sources = "\n".join(f"{ev['source_id']}: [{ev['citation_number']}, p.{ev['page']}] {ev['excerpt'][:320]}" for ev in evidence)
    assessed = llm.generate_structured(
        prompt_template("domain") + "\n" + "Evaluate each of D1–D7 for each technology separately. Return 14 distinct items; verdict only 적합/조건부/제약/근거 부족. "
        "If the underlying answer is insufficient, use 근거 부족, not assumed performance. Cite only listed IDs. "
        "Do NOT award winner or aggregate numeric score.\n" + repr(research) + "\nSource excerpts:\n" + sources
        + rework_note(feedback),
        DomainAssessment,
    )
    valid_ids = {ev["source_id"] for ev in evidence}
    items = downgrade_single_document(normalize_items([item.model_dump() for item in assessed.items], evidence), evidence)
    expected = {(tech, f"{code} {title}") for tech in ("mla", "itme") for code, title, _ in DIMENSIONS}
    actual = {(item["technology"], item["dimension"]) for item in items}
    if actual != expected or len(items) != 14:
        missing.append("domain: D1~D7 각 기술의 14개 평가 결과 불완전")
    for item in items:
        if any(cid not in valid_ids for cid in item["cited_ids"]):
            missing.append(f"domain/{item['technology']}: {item['dimension']} 존재하지 않는 문헌 출처")
        if item["verdict"] != "근거 부족" and not item["cited_ids"]:
            missing.append(f"domain/{item['technology']}: {item['dimension']} 판정 근거 인용 없음")
    optional.extend(assessed.missing)
    missing = list(dict.fromkeys(missing))
    cached = {f"{t[0]} {t[1]} {t[2]}: {t[3]}": answer for t, answer in zip(tasks, answers)}
    return {"perspectives": {"domain": {**assessed.model_dump(), "items": items, "attempt": attempt,
                                        "llm_sufficient": assessed.sufficient,
                                        "sufficient": not missing, "missing": missing,
                                        "missing_optional": list(dict.fromkeys(optional))}},
            "cache_keys": {"domain": rag_cache.save(state["trace_id"], "domain", cached)},
            "evidence": evidence}
