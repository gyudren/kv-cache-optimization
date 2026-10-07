"""관측성: trace_id, 결정 로그(JSONL), LangGraph·LangSmith 실행 설정.

trace_id 하나를 체크포인트 thread_id, LangSmith run metadata, 결정 로그 파일명에 같이 쓴다.
"""
from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .config import AGENT_CONCURRENCY, LANGSMITH_RUN_NAME, LANGSMITH_TAGS


def new_trace_id() -> str:
    return str(uuid.uuid4())


def decision_log_path(trace_id: str) -> Path:
    return Path(os.getenv("OUTPUT_DIR", "outputs")) / f"decisions_{trace_id}.jsonl"


def log_decision(trace_id: str, step: int, node: str, decision: str, reason: str, **extra) -> dict:
    """결정 1건을 JSONL에 덧붙이고, State의 last_decision에 넣을 레코드를 돌려준다."""
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
    """LangGraph 실행 설정. recursion_limit은 build_graph가 Policy로 정하므로 여기 넣지 않는다."""
    return {
        "configurable": {"thread_id": trace_id},
        "max_concurrency": AGENT_CONCURRENCY,
        "run_name": LANGSMITH_RUN_NAME,
        "tags": list(LANGSMITH_TAGS),
        "metadata": {"trace_id": trace_id, "project": os.getenv("LANGSMITH_PROJECT", "")},
    }
