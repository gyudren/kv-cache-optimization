"""단일 Supervisor 노드와 라우터.

그래프에는 `add_conditional_edges("supervisor", route, ...)` 하나만 있다. supervisor 노드가
policy.decide로 다음 노드를 정해 State(next_agents)에 쓰고, route는 그 값만 읽어 경로를 만든다.
관점 에이전트는 `Send`로 동적 fan-out하므로 근거가 부족한 관점 1개만 다시 보낼 수도, 4개를 한 번에 보낼 수도 있다.
할당은 한 번에 하지만 실행은 run_config의 max_concurrency(기본 1)에 따라 하나씩 순차로 한다(메모리 상한).
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


# 관점 에이전트가 읽는 필드만 Send로 넘긴다. State 전체를 넘기면 체크포인트의 대기 작업(Send)마다
# evidence·보고서까지 복제되어 체크포인트가 fan-out 수만큼 커진다.
SEND_KEYS = ("user_query", "trace_id", "retry_counts", "followup_counts", "feedback", "step_count")


def route(state: dict):
    """State의 next_agents만으로 경로를 정한다(라우터 안에서 새 판단을 하지 않는다)."""
    targets = state.get("next_agents") or []
    if not targets or END_NODE in targets:
        return END
    fanout = [name for name in targets if name in PERSPECTIVES]
    if fanout:
        payload = {key: state[key] for key in SEND_KEYS if key in state}
        return [Send(name, payload) for name in fanout]
    return targets[0]
