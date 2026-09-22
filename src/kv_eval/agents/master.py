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
    sufficient = bool(state.get("tech_result", {}).get("sufficient"))
    counts = dict(state["retry_counts"])
    feedback = dict(state.get("review_feedback", {}))
    route = "dispatch"
    if not sufficient and counts["tech"] < RETRY_LIMITS["tech"]:
        counts["tech"] += 1
        route = "technology"
        feedback["tech"] = {"missing": state.get("tech_result", {}).get("missing", []),
                            "rewritten_queries": state.get("tech_result", {}).get("missing", [])}
    return {"retry_counts": counts, "review_feedback": feedback,
            "phase": "tech_research" if route == "technology" else "parallel_eval",
            "next_agents": [] if route == "technology" else list(PERSPECTIVES),
            "logs": [{"node": "master_tech_gate", "gate": sufficient, "result": route,
                      "attempt": counts["tech"]}]}


def route_tech(state: dict) -> str:
    return "technology" if state["phase"] == "tech_research" else "master_dispatch"


def master_dispatch_node(state: dict) -> dict:
    selected = state.get("next_agents", [])
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(PERSPECTIVES):
        raise ValueError("Master dispatch must select unique eligible perspective agents")
    return {"logs": [{"node": "master_dispatch", "result": ",".join(selected),
                      "attempt": 0}]}


def master_join_node(state: dict) -> dict:
    # After source reassignment from synthesis, return directly to synthesis;
    # source retry counts were already increased by master_synthesis_gate.
    if state["phase"] == "synthesis":
        return {"next_agents": [], "logs": [{"node": "master_join", "result": "synthesis_return"}]}
    counts = dict(state["retry_counts"])
    feedback = dict(state.get("review_feedback", {}))
    retry = []
    for agent in PERSPECTIVES:
        result = state.get(RESULT_KEY[agent], {})
        if not result.get("sufficient") and counts[agent] < RETRY_LIMITS[agent]:
            counts[agent] += 1
            feedback[agent] = {"missing": result.get("missing", []),
                               "rewritten_queries": result.get("missing", [])}
            retry.append(agent)
    return {"retry_counts": counts, "review_feedback": feedback, "next_agents": retry,
            "phase": "parallel_eval" if retry else "synthesis",
            "logs": [{"node": "master_join", "result": "retry" if retry else "synthesis",
                      "gate": not bool(retry), "attempt": sum(counts[a] for a in PERSPECTIVES)}]}


def route_join(state: dict) -> str:
    return "master_dispatch" if state["phase"] == "parallel_eval" else "synthesis"


def master_synthesis_gate_node(state: dict) -> dict:
    synthesis = state.get("synthesis_result", {})
    counts = dict(state["retry_counts"])
    feedback = dict(state.get("review_feedback", {}))
    # Revisit missing source perspectives before retrying synthesis wording.
    source_agents = list(dict.fromkeys(a for a in synthesis.get("needs_source_agents", [])
                                       if a in RESULT_KEY and a != "tech" and counts[a] < RETRY_LIMITS[a]))
    # Technical findings cannot be assigned to the perspective-only dispatch.
    # Escalate any technical gap to the report instead of silently reassigning the wrong agent.
    if source_agents:
        for agent in source_agents:
            counts[agent] += 1
            feedback[agent] = {"missing": synthesis.get("evidence_gaps", []),
                               "rewritten_queries": synthesis.get("evidence_gaps", [])}
        route = "master_dispatch"
        phase = "synthesis"
    elif synthesis.get("needs_revision") and counts["synthesis"] < RETRY_LIMITS["synthesis"]:
        counts["synthesis"] += 1
        feedback["synthesis"] = {"issues": synthesis.get("evidence_gaps", [])}
        route = "synthesis"
        phase = "synthesis"
    else:
        route = "report"
        phase = "report"
    return {"retry_counts": counts, "review_feedback": feedback,
            "next_agents": source_agents if source_agents else [], "phase": phase,
            "logs": [{"node": "master_synthesis_gate", "result": route,
                      "gate": not bool(synthesis.get("evidence_gaps", [])),
                      "attempt": counts["synthesis"]}]}


def route_synthesis(state: dict) -> str:
    if state["phase"] == "report":
        return "report"
    return "master_dispatch" if state["next_agents"] else "synthesis"


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
