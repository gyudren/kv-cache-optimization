"""Supervisor 노드와 라우터.

supervisor_node가 policy.decide 결과를 next_agents에 쓰고, route는 그 값만 읽는다.
관점 에이전트는 Send로 필요한 것만 골라 보낸다.
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


# 관점 에이전트가 읽는 필드만 넘긴다. State 전체를 넘기면 대기 중인 Send마다 체크포인트에 복제된다.
SEND_KEYS = ("user_query", "trace_id", "retry_counts", "followup_counts", "reinvestigate_counts", "feedback",
             "step_count")


def route(state: dict):
    """next_agents만 보고 경로를 정한다. 판단은 policy.decide에서 끝난다."""
    targets = state.get("next_agents") or []
    if not targets or END_NODE in targets:
        return END
    fanout = [name for name in targets if name in PERSPECTIVES]
    if fanout:
        payload = {key: state[key] for key in SEND_KEYS if key in state}
        return [Send(name, payload) for name in fanout]
    return targets[0]
