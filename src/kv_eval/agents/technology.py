"""Technical research: 6 paper questions per selected technology plus 1 HW baseline."""
from __future__ import annotations
from typing import Any
from ..prompts import prompt_template
from ..schemas import TechnologyAssessment

TECH_QUESTIONS = [
    "What problem and scope does the proposed KV cache approach address?",
    "How does the key-value compression or memory hierarchy mechanism work?",
    "What implementation and experimental setup is documented?",
    "What quantitative memory and throughput/latency outcomes are reported and under what conditions?",
    "What limitations and prerequisites does the paper state?",
    "What evidence exists for implementation, validation, and technology readiness (TRL)?",
]
BASELINE_QUESTION = "How do InfiniGen and CXL-PNM differ from ITME in memory expansion mechanism, measured trade-offs, and stated limitations?"


def technology_node(state: dict, rag: Any, llm: Any) -> dict:
    findings: dict = {}
    evidence: list[dict] = []
    missing: list[str] = []
    attempt = state["retry_counts"]["tech"]
    feedback = state.get("review_feedback", {}).get("tech", {})
    for tech in ("mla", "itme"):
        answers = []
        for question in TECH_QUESTIONS:
            response = rag.rag_answer(f"[{tech}] {question}", tech, feedback)
            answers.append(response["answer"])
            missing.extend(response["missing"])
            for item in response["evidence"]:
                evidence.append({**item, "agent": "tech", "attempt": attempt})
        findings[tech] = {"answers": answers}
    baseline = rag.rag_answer(BASELINE_QUESTION, "itme_baseline", feedback)
    findings["itme_baseline"] = baseline["answer"]
    missing.extend(baseline["missing"])
    evidence.extend({**item, "agent": "tech", "attempt": attempt} for item in baseline["evidence"])
    refs = "\n".join(f"[{ev['citation_number']}, p.{ev['page']}] {ev['claim']}: {ev['excerpt'][:300]}" for ev in evidence)
    result = llm.generate_structured(
        prompt_template("technology") + "\n" + "Summarize MLA and ITME technical scope, reported results, limitations and separately assessed TRL from ONLY the research excerpts. "
        "Do not confuse product-family deployment with adoption of the exact architecture. "
        "TRL ranges: 1-3 published concept, 4-6 implementation/validation, 7-9 documented operational adoption. "
        "No unsupported claims.\n" + "\n".join(f"{k}: {v}" for k, v in findings.items()) + "\nSources:\n" + refs,
        TechnologyAssessment,
    )
    missing.extend(result.missing)
    if set(result.trl) != {"mla", "itme"} or set(result.trl_basis) != {"mla", "itme"}:
        missing.append("MLA/ITME의 독립적 TRL 근거 미기재")
    return {"tech_result": {**result.model_dump(), "details": findings,
                            "sufficient": result.sufficient and not bool(missing), "missing": list(dict.fromkeys(missing))},
            "evidence": evidence,
            "logs": [{"node": "technology", "attempt": attempt, "result": "complete", "gate": not bool(missing)}]}
