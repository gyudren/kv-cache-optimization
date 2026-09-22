"""D-1 State 및 reducer.

병렬 Agent가 같은 key를 동시에 쓰지 않도록 관점별 결과를 전용 key로 분리하고,
공통 누적분(evidence, logs)만 reducer를 쓴다.
단계 값·재시도 한도 같은 설계 고정값은 config에 둔다.
"""

import hashlib
import operator
from typing import Annotated, TypedDict


class EvalState(TypedDict, total=False):
    user_query: str
    phase: str
    next_agents: list[str]

    # 관점별 결과 — 각 Agent 전용 key (병렬 충돌 방지)
    tech_result: dict
    market_result: dict
    stakeholder_result: dict
    domain_result: dict

    # 공통 누적 — reducer
    evidence: Annotated[list, operator.add]
    logs: Annotated[list, operator.add]

    review_feedback: dict
    synthesis_result: dict
    report_draft: str

    # 작성 주체는 Master. 병렬 노드가 쓰면 동시 갱신으로 InvalidUpdateError가 난다.
    retry_counts: dict
    status: str


def source_id(url: str) -> str:
    """URL 기반 안정 식별자. 전 Agent가 같은 함수를 써야 종합 단계 중복 제거가 성립한다."""
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def to_evidence(hit: dict, *, agent: str, attempt: int, **fields) -> dict:
    """검색 결과 1건을 D-1 evidence 항목으로 변환한다. 필수 3필드를 여기서 강제한다."""
    url = hit.get("url", "")
    return {
        "source_id": source_id(url),
        "agent": agent,
        "attempt": attempt,
        "title": hit.get("title", ""),
        "url": url,
        "content": hit.get("content", ""),
        "score": hit.get("score"),
        **fields,
    }


def initial_state(user_query: str) -> EvalState:
    return {
        "user_query": user_query,
        "phase": "init",
        "next_agents": [],
        "evidence": [],
        "logs": [],
        "review_feedback": {},
        "retry_counts": {},
        "status": "running",
    }
