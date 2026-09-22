"""7 shared domain dimensions, grounded in each selected primary paper only."""
from __future__ import annotations
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
    expected = {(tech, f"{code} {title}") for tech in ("mla", "itme") for code, title, _ in DIMENSIONS}
    actual = {(item.technology, item.dimension) for item in assessed.items}
    if actual != expected or len(assessed.items) != 14:
        missing.append("D1~D7 각 기술의 14개 평가 결과 불완전")
    for item in assessed.items:
        if any(cid not in valid_ids for cid in item.cited_ids):
            missing.append(f"{item.technology}/{item.dimension}: 존재하지 않는 문헌 출처")
    missing.extend(assessed.missing)
    return {"domain_result": {**assessed.model_dump(), "sufficient": assessed.sufficient and not bool(missing),
                              "missing": list(dict.fromkeys(missing))},
            "evidence": evidence,
            "logs": [{"node": "domain", "attempt": attempt, "result": "complete", "gate": not bool(missing)}]}
