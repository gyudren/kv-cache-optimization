"""D-1 state contract; only Master modifies control-flow keys."""
from __future__ import annotations
import operator
from typing import Annotated, Any, Literal, TypedDict
from .config import RETRY_LIMITS

class GraphState(TypedDict):
    user_query: str
    phase: Literal["init", "tech_research", "parallel_eval", "synthesis", "report", "done"]
    next_agents: list[str]
    tech_result: dict[str, Any]
    market_result: dict[str, Any]
    stakeholder_result: dict[str, Any]
    domain_result: dict[str, Any]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    logs: Annotated[list[dict[str, Any]], operator.add]
    review_feedback: dict[str, Any]
    synthesis_result: dict[str, Any]
    report_draft: str
    retry_counts: dict[str, int]
    status: Literal["running", "completed", "failed"]

RESULT_KEY = {"tech": "tech_result", "market": "market_result", "stakeholder": "stakeholder_result", "domain": "domain_result"}


def initial_state(query: str) -> GraphState:
    return {
        "user_query": query,
        "phase": "init", "next_agents": [],
        "tech_result": {}, "market_result": {}, "stakeholder_result": {}, "domain_result": {},
        "evidence": [], "logs": [], "review_feedback": {}, "synthesis_result": {},
        "report_draft": "", "retry_counts": {name: 0 for name in RETRY_LIMITS}, "status": "running",
    }


def deduplicate_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, object]] = set()
    out = []
    for ev in items:
        key = (ev.get("source_id", ""), ev.get("claim", ""), ev.get("page") or ev.get("url"))
        if key not in seen:
            seen.add(key)
            out.append(ev)
    return out


def prompt_view(result: dict) -> dict:
    """LLM 프롬프트에 넣을 결과 사본. 재시도용 내부 캐시(rag_cache)는 근거 원문 전체라 제외한다."""
    return {k: v for k, v in (result or {}).items() if k != "rag_cache"}
