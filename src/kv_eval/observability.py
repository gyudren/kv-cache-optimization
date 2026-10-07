"""관측성: trace_id 상관 키, 결정 로그(JSONL), LangSmith 실행 설정.

하나의 trace_id(uuid4)가 세 저장소를 잇는다.
- LangGraph 체크포인트: config["configurable"]["thread_id"]
- LangSmith: run metadata["trace_id"] (+ run_name, tags=["pattern:supervisor"])
- 결정 로그: outputs/decisions_{trace_id}.jsonl

결정 로그 본문은 State에 쌓지 않는다(체크포인트마다 전체 이력이 복제되므로). State에는 마지막 1건만 남긴다.
"""
from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .config import LANGSMITH_RUN_NAME, LANGSMITH_TAGS


def new_trace_id() -> str:
    return str(uuid.uuid4())


def decision_log_path(trace_id: str) -> Path:
    return Path(os.getenv("OUTPUT_DIR", "outputs")) / f"decisions_{trace_id}.jsonl"


def log_decision(trace_id: str, step: int, node: str, decision: str, reason: str, **extra) -> dict:
    """결정 1건을 JSONL에 append하고 같은 레코드를 돌려준다(State의 last_decision용)."""
    record = {"trace_id": trace_id, "step": step, "node": node, "decision": decision, "reason": reason,
              "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}
    path = decision_log_path(trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_decisions(trace_id: str) -> list[dict]:
    path = decision_log_path(trace_id)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def langsmith_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY", "").strip())


def run_config(trace_id: str) -> dict:
    """LangGraph 실행 설정. LANGSMITH_TRACING=true면 LangSmith가 run_name·tags·metadata를 그대로 기록한다.

    recursion_limit은 여기서 정하지 않는다. build_graph가 그 그래프의 Policy.max_steps로 계산해
    그래프 기본 설정에 넣으므로 Policy와 상한이 어긋날 수 없다(단일 출처).
    """
    return {
        "configurable": {"thread_id": trace_id},
        "run_name": LANGSMITH_RUN_NAME,
        "tags": list(LANGSMITH_TAGS),
        "metadata": {"trace_id": trace_id, "project": os.getenv("LANGSMITH_PROJECT", "")},
    }
