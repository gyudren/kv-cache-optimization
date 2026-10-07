"""에이전트 실패 처리 래퍼.

성공한 노드의 evidence는 여기서 원문을 evidence_store로 옮긴다.

예외가 그래프 밖으로 나가면 실행 전체가 죽고 보고서가 남지 않는다. 래퍼가 예외를 잡아
node_status[name]=failed, last_error[name]=요약으로 바꾸면 Supervisor가 재시도 한도 안에서
다시 보내거나, 한도를 넘으면 제외하고 근거 공백으로 기록한다.
"""
from __future__ import annotations
from typing import Callable
from langgraph.errors import GraphBubbleUp
from .. import evidence_store
from ..config import PERSPECTIVES


def guarded(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    def node(state: dict) -> dict:
        try:
            update = fn(state) or {}
        except GraphBubbleUp:
            raise  # interrupt 등 LangGraph 제어 신호는 그대로 올린다
        except Exception as exc:  # noqa: BLE001 - 에이전트 실패를 State로 바꾸는 경계
            return {"node_status": {name: "failed"},
                    "last_error": {name: f"{type(exc).__name__}: {exc}"[:300]}}
        if name in PERSPECTIVES and name in (update.get("perspectives") or {}):
            # 충분성은 "이번 시도"가 수집한 출처로만 판단한다(이전 시도 evidence가 누적돼 통과되지 않게).
            result = {**update["perspectives"][name],
                      "source_units": evidence_store.source_units(update.get("evidence") or [])}
            update = {**update, "perspectives": {**update["perspectives"], name: result}}
        if update.get("evidence") and state.get("trace_id"):
            # 노드 경계에서 발췌 원문을 디스크로 옮기고 State에는 축약본·참조만 넘긴다(체크포인트 크기 제한).
            update = {**update, "evidence": evidence_store.offload(state["trace_id"], name, update["evidence"])}
        return {**update, "node_status": {name: "done"}, "last_error": {name: ""}}

    node.__name__ = f"{name}_node"
    return node
