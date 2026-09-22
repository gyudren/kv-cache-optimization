"""Master role: stage state, gating, selectively retrying, and report review."""
from __future__ import annotations
from typing import Any
from ..config import RETRY_LIMITS
from ..prompts import prompt_template
from ..reporting.sections import validate_report, citeable_evidence
from ..schemas import ReportAssessment
from ..state import RESULT_KEY

PERSPECTIVES = ("market", "stakeholder", "domain")


def master_init_node(state: dict) -> dict:
    return {"phase": "tech_research", "next_agents": [], "review_feedback": {},
            "retry_counts": {name: 0 for name in RETRY_LIMITS}, "status": "running",
            "logs": [{"node": "master_init", "attempt": 0, "result": "initialized"}]}


def master_tech_gate_node(state: dict) -> dict:
    """설계 D-2 SUP_TECH: 기술·TRL 근거가 충분한지만 판정하고 경로를 정한다.

    부족 항목 지정과 질의 재작성은 별도 노드(master_query_rewrite)가 수행한다.
    """
    sufficient = bool(state.get("tech_result", {}).get("sufficient"))
    counts = state["retry_counts"]
    rewrite = not sufficient and counts["tech"] < RETRY_LIMITS["tech"]
    return {"phase": "tech_research" if rewrite else "parallel_eval",
            "next_agents": [] if rewrite else list(PERSPECTIVES),
            "logs": [{"node": "master_tech_gate", "gate": sufficient,
                      "result": "query_rewrite" if rewrite else "dispatch",
                      "attempt": counts["tech"]}]}


def route_tech(state: dict) -> str:
    return "master_query_rewrite" if state["phase"] == "tech_research" else "master_dispatch"


def master_query_rewrite_node(state: dict) -> dict:
    """설계 D-2 QUERY_REWRITE: 부족 항목을 지정하고 재검색 질의를 넘긴다(최대 2회)."""
    counts = dict(state["retry_counts"])
    counts["tech"] += 1
    missing = state.get("tech_result", {}).get("missing", [])
    feedback = dict(state.get("review_feedback", {}))
    feedback["tech"] = {"missing": missing, "rewritten_queries": missing}
    return {"retry_counts": counts, "review_feedback": feedback,
            "logs": [{"node": "master_query_rewrite", "gate": False, "result": "rewritten",
                      "attempt": counts["tech"]}]}


def master_dispatch_node(state: dict) -> dict:
    selected = state.get("next_agents", [])
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(PERSPECTIVES):
        raise ValueError("Master dispatch must select unique eligible perspective agents")
    return {"logs": [{"node": "master_dispatch", "result": ",".join(selected),
                      "attempt": 0}]}


def master_join_node(state: dict) -> dict:
    # 설계 D-2 RESULT_GATE: 결과 수집 후 "모든 관점 결과가 완료되었는가"만 판정한다.
    # 재할당 준비(횟수 증가·부족 항목 전달)는 master_retry 노드가 수행한다.
    counts = state["retry_counts"]
    retry = [agent for agent in PERSPECTIVES
             if not state.get(RESULT_KEY[agent], {}).get("sufficient")
             and counts[agent] < RETRY_LIMITS[agent]]
    return {"next_agents": retry,
            "phase": "parallel_eval" if retry else "synthesis",
            "logs": [{"node": "master_join", "result": "retry" if retry else "synthesis",
                      "gate": not bool(retry), "attempt": sum(counts[a] for a in PERSPECTIVES)}]}


def route_join(state: dict) -> str:
    return "master_retry" if state["phase"] == "parallel_eval" else "synthesis"


def master_retry_node(state: dict) -> dict:
    """설계 D-2 RETRY: 근거가 부족한 Agent만 골라 부족 항목과 함께 다시 할당한다."""
    counts = dict(state["retry_counts"])
    feedback = dict(state.get("review_feedback", {}))
    targets = list(state.get("next_agents", []))
    for agent in targets:
        counts[agent] += 1
        missing = state.get(RESULT_KEY[agent], {}).get("missing", [])
        feedback[agent] = {"missing": missing, "rewritten_queries": missing}
    return {"retry_counts": counts, "review_feedback": feedback,
            "logs": [{"node": "master_retry", "gate": False, "result": ",".join(targets),
                      "attempt": sum(counts[a] for a in PERSPECTIVES)}]}


def master_synthesis_gate_node(state: dict) -> dict:
    """설계 D-2 SUP_SYNTHESIS: 일치·상충·근거 공백이 정리됐는지 판정한다.

    경로는 설계대로 "부족(최대 1회) → 종합 재실행" 또는 "충분 → 보고서" 둘뿐이다.
    종합 단계에서 새로 발견된 근거 부족은 관점 Agent를 다시 부르지 않고(설계에 없는 경로)
    근거 공백으로 기록해 보고서의 한계점으로 넘긴다. 관점별 근거 재수집은 앞단의
    RESULT_GATE → RETRY 루프가 담당한다.
    """
    synthesis = dict(state.get("synthesis_result", {}))
    counts = dict(state["retry_counts"])
    feedback = dict(state.get("review_feedback", {}))

    gaps = list(synthesis.get("evidence_gaps", []))
    unreachable = [agent for agent in synthesis.get("needs_source_agents", []) if agent in RESULT_KEY]
    if unreachable:
        gaps = list(dict.fromkeys(gaps + [f"{agent}: 추가 근거 필요(재검색 한도 내에서 확보되지 않음)"
                                          for agent in unreachable]))
        synthesis["evidence_gaps"] = gaps

    if synthesis.get("needs_revision") and counts["synthesis"] < RETRY_LIMITS["synthesis"]:
        counts["synthesis"] += 1
        feedback["synthesis"] = {"issues": gaps}
        route, phase = "synthesis", "synthesis"
    else:
        route, phase = "report", "report"
    return {"retry_counts": counts, "review_feedback": feedback, "synthesis_result": synthesis,
            "next_agents": [], "phase": phase,
            "logs": [{"node": "master_synthesis_gate", "result": route,
                      "gate": not bool(gaps), "attempt": counts["synthesis"]}]}


def route_synthesis(state: dict) -> str:
    return "report" if state["phase"] == "report" else "synthesis"


def master_report_gate_node(state: dict, llm: Any) -> dict:
    deterministic = validate_report(state["report_draft"], state["evidence"])
    # Judge is same fixed GPT-5.6 Sol; no added standalone evaluation Agent.
    judged = llm.generate_structured(
        prompt_template("master", "validator") + "\n"
        "Check the report against these exact requirements: headings SUMMARY, 1~7 and REFERENCE; "
        "neutral perspective comparison; unsupported claims and missing citations must be flagged. "
        "This is an internal report gate, not a comparative technology score. "
        "Do not overrule deterministic citation-validation failures.\n"
        f"Deterministic issues: {deterministic['issues']}\n"
        f"Verified evidence snippets: {repr([{'citation': ev['citation'], 'excerpt': ev.get('excerpt', '')[:900], 'claim': ev.get('claim', '')} for ev in citeable_evidence(state['evidence'])])[:65000]}\n"
        f"Report:\n{state['report_draft'][:55000]}",
        ReportAssessment,
    )
    issues = list(dict.fromkeys(deterministic["issues"] + judged.issues))
    passed = deterministic["passed"] and judged.passed and not issues
    counts = dict(state["retry_counts"])
    feedback = dict(state.get("review_feedback", {}))
    if not passed:
        feedback["report"] = {"issues": issues or ["Judge verification failed"]}
    if not passed and counts["report"] < RETRY_LIMITS["report"]:
        counts["report"] += 1
        return {"retry_counts": counts, "review_feedback": feedback, "phase": "report",
                "logs": [{"node": "master_report_gate", "gate": False, "result": "revise",
                          "attempt": counts["report"], "reason": "; ".join(issues)[:700]}]}
    return {"retry_counts": counts, "review_feedback": feedback,
            "phase": "done", "status": "completed",
            "logs": [{"node": "master_report_gate", "gate": passed,
                      "result": "verified" if passed else "unverified_exhausted",
                      "attempt": counts["report"], "reason": "; ".join(issues)[:700]}]}


def route_report(state: dict) -> str:
    return "report" if state["phase"] == "report" else "end"
