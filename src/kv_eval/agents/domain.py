"""7 shared domain dimensions, grounded in each selected primary paper only."""
from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from ..config import MAX_PARALLEL_QUESTIONS
from ..prompts import prompt_template
from ..schemas import DomainAssessment

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
    """LLM 출력의 평가항목명·인용을 검증 가능한 형태로 맞춘다.

    LLM은 항목명을 "MLA D1 워크로드 수용 능력"처럼 기술명을 붙이거나 조금 바꿔 쓰고,
    인용도 source_id 대신 "[1, p.7]" 문자열로 돌려주는 경우가 많다. 이를 그대로 비교하면
    판정이 맞아도 매번 "평가 결과 불완전·존재하지 않는 출처"로 실패해 재시도만 소진된다.
    항목은 D-코드로, 인용은 실제 수집된 Evidence의 source_id로 정규화한다(없는 인용은 남겨 검증에서 걸리게 한다).
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


def domain_node(state: dict, rag: Any, llm: Any) -> dict:
    attempt = state["retry_counts"]["domain"]
    feedback = state.get("review_feedback", {}).get("domain", {})
    research = []
    evidence = []
    missing = []
    tasks = [(tech, code, title, question)
             for tech in ("mla", "itme") for code, title, question in DIMENSIONS]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_QUESTIONS) as pool:
        answers = list(pool.map(
            lambda t: rag.rag_answer(f"{t[0]} {t[1]} {t[2]}: {t[3]}", t[0], feedback), tasks))
    for (tech, code, title, question), answer in zip(tasks, answers):
        research.append({"technology": tech, "dimension": f"{code} {title}", "question": question,
                         "answer": answer["answer"], "sufficient": answer["sufficient"]})
        missing.extend(answer["missing"])
        evidence.extend({**item, "agent": "domain", "attempt": attempt} for item in answer["evidence"])
    sources = "\n".join(f"{ev['source_id']}: [{ev['citation_number']}, p.{ev['page']}] {ev['excerpt'][:320]}" for ev in evidence)
    assessed = llm.generate_structured(
        prompt_template("domain") + "\n" + "Evaluate each of D1–D7 for each technology separately. Return 14 distinct items; verdict only 적합/조건부/제약/근거 부족. "
        "If the underlying answer is insufficient, use 근거 부족, not assumed performance. Cite only listed IDs. "
        "Do NOT award winner or aggregate numeric score.\n" + repr(research) + "\nSource excerpts:\n" + sources,
        DomainAssessment,
    )
    valid_ids = {ev["source_id"] for ev in evidence}
    items = normalize_items([item.model_dump() for item in assessed.items], evidence)
    expected = {(tech, f"{code} {title}") for tech in ("mla", "itme") for code, title, _ in DIMENSIONS}
    actual = {(item["technology"], item["dimension"]) for item in items}
    if actual != expected or len(items) != 14:
        missing.append("D1~D7 각 기술의 14개 평가 결과 불완전")
    for item in items:
        if any(cid not in valid_ids for cid in item["cited_ids"]):
            missing.append(f"{item['technology']}/{item['dimension']}: 존재하지 않는 문헌 출처")
        if item["verdict"] != "근거 부족" and not item["cited_ids"]:
            missing.append(f"{item['technology']}/{item['dimension']}: 판정 근거 인용 없음")
    missing.extend(assessed.missing)
    return {"domain_result": {**assessed.model_dump(), "items": items,
                              "sufficient": assessed.sufficient and not bool(missing),
                              "missing": list(dict.fromkeys(missing))},
            "evidence": evidence,
            "logs": [{"node": "domain", "attempt": attempt, "result": "complete", "gate": not bool(missing)}]}
