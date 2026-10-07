#!/usr/bin/env python3
"""State·체크포인트 크기 실측 (API 키 불필요).

1) Fake LLM·Web·RAG로 그래프를 SqliteSaver와 함께 1회 실행하고(발췌 길이는 실데이터와 비슷하게
   논문 1,200자·웹 1,000자), 최종 State JSON 크기와 체크포인트 누적 크기를 잰다.
2) 이전 실제 실행의 State(outputs/legacy_rag/final_state.json)에서 evidence 필드가 현재 저장 방식으로
   얼마가 되는지 계산한다.

    python scripts/measure_state_size.py [--scenario normal|rework]
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def _size(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False, default=str).encode())


def measure_fake(scenario: str) -> dict:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from fakes import FakeLLM, FakeRAG, FakeWeb
    from kv_eval.graph import build_graph
    from kv_eval.observability import new_trace_id, run_config
    from kv_eval.state import initial_state
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(OUTPUT_DIR=f"{tmp}/out", RAG_CACHE_DIR=f"{tmp}/cache",
                          MANIFEST_PATH=str(ROOT / "data" / "manifest.json"))
        llm = FakeLLM() if scenario == "normal" else FakeLLM(insufficient={"market": 1}, one_sided={"stakeholder": 1})
        conn = sqlite3.connect(f"{tmp}/ckpt.sqlite", check_same_thread=False)
        trace_id = new_trace_id()
        graph = build_graph(FakeRAG(), FakeWeb(), llm, checkpointer=SqliteSaver(conn))
        state = graph.invoke(initial_state("q", trace_id), config=run_config(trace_id))
        rows, blob = conn.execute("select count(*), sum(length(checkpoint)) from checkpoints").fetchone()
        writes = conn.execute("select coalesce(sum(length(value)),0) from writes").fetchone()[0]
        disk = sum(p.stat().st_size for p in Path(f"{tmp}/cache").rglob("*") if p.is_file())
        return {"scenario": scenario, "status": state["status"], "evidence_items": len(state["evidence"]),
                "state_json_bytes": _size(state), "evidence_bytes": _size(state["evidence"]),
                "checkpoints": rows, "checkpoint_bytes": blob, "pending_write_bytes": writes,
                "external_store_bytes": disk}


def measure_legacy() -> dict | None:
    from kv_eval.config import STATE_EXCERPT_CHARS
    path = ROOT / "outputs" / "legacy_rag" / "final_state.json"
    if not path.is_file():
        return None
    evidence = json.loads(path.read_text(encoding="utf-8"))["evidence"]
    compact = [{**ev, "excerpt": (ev.get("excerpt") or "")[:STATE_EXCERPT_CHARS], "excerpt_ref": "x" * 52}
               for ev in evidence]
    return {"items": len(evidence), "full_evidence_bytes": _size(evidence), "compact_evidence_bytes": _size(compact)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["normal", "rework"], default="rework")
    args = parser.parse_args()
    print(json.dumps({"fake_run": measure_fake(args.scenario), "legacy_real_run": measure_legacy()},
                     ensure_ascii=False, indent=2))
