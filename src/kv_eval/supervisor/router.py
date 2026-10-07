"""단일 Supervisor 노드와 라우터.

그래프에는 `add_conditional_edges("supervisor", route, ...)` 하나만 있다. supervisor 노드가
policy.decide로 다음 노드를 정해 State(next_agents)에 쓰고, route는 그 값만 읽어 경로를 만든다.
관점 에이전트는 `Send`로 동적 fan-out하므로 근거가 부족한 관점 1개만 다시 보낼 수도, 4개를 동시에 보낼 수도 있다.
"""
from __future__ import annotations
from langgraph.graph import END
from langgraph.types import Send
from ..config import PERSPECTIVES
from ..observability import log_decision
from .policy import END_NODE, Policy, decide


def supervisor_node(state: dict, policy: Policy | None = None) -> dict:
    decision = decide(state, policy)
    step = int(state.get("step_count", 0)) + 1
    record = log_decision(state["trace_id"], step, "supervisor", decision.decision, decision.reason,
                          targets=decision.targets)
    return {**decision.updates, "step_count": step, "next_agents": decision.targets, "last_decision": record}


def route(state: dict):
    """State의 next_agents만으로 경로를 정한다(라우터 안에서 새 판단을 하지 않는다)."""
    targets = state.get("next_agents") or []
    if not targets or END_NODE in targets:
        return END
    fanout = [name for name in targets if name in PERSPECTIVES]
    if fanout:
        return [Send(name, state) for name in fanout]
    return targets[0]
