"""작업 노드 래퍼. 예외가 실행 전체를 멈추지 않도록 node_status=failed, last_error로 바꾼다.

에이전트 본문은 블로킹 I/O라 asyncio.to_thread로 돌린다. to_thread가 contextvars를 복사하므로
에이전트 안의 LLM·웹 호출도 그래프 run의 자식으로 기록된다.
"""
from __future__ import annotations
import asyncio
from typing import Awaitable, Callable
from langgraph.errors import GraphBubbleUp
from .. import evidence_store
from ..config import PERSPECTIVES


def guarded(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], Awaitable[dict]]:
    async def node(state: dict) -> dict:
        try:
            update = (await asyncio.to_thread(fn, state)) or {}
        except GraphBubbleUp:
            raise  # interrupt 등 LangGraph 제어 신호는 그대로 올린다
        except Exception as exc:  # noqa: BLE001 - 에이전트 실패를 State로 바꾸는 경계
            return {"node_status": {name: "failed"},
                    "last_error": {name: f"{type(exc).__name__}: {exc}"[:300]}}
        if name in PERSPECTIVES and name in (update.get("perspectives") or {}):
            # 이전 시도 근거로 충분성이 통과되지 않도록 이번 시도의 출처만 센다.
            result = {**update["perspectives"][name],
                      "source_units": evidence_store.source_units(update.get("evidence") or [])}
            update = {**update, "perspectives": {**update["perspectives"], name: result}}
        if update.get("evidence") and state.get("trace_id"):
            # 체크포인트가 커지지 않게 발췌 원문은 디스크로 옮기고 State에는 축약본과 참조만 남긴다.
            update = {**update, "evidence": evidence_store.offload(state["trace_id"], name, update["evidence"])}
        return {**update, "node_status": {name: "done"}, "last_error": {name: ""}}

    node.__name__ = f"{name}_node"
    return node
