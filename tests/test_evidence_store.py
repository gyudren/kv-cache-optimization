"""지속성 비용(D-05): State에는 축약 발췌만, 원문은 디스크 저장소에서 되살려 보고서·평가에 쓴다."""
from __future__ import annotations
import re

from fakes import FakeLLM
from kv_eval import evidence_store
from kv_eval.config import STATE_EXCERPT_CHARS
from kv_eval.evaluation.quality import cited_snippets


def test_offload_and_hydrate_roundtrip():
    full = "가" * 1200
    ev = [{"source_id": "s1", "claim": "c", "page": 3, "excerpt": full}, {"source_id": "s2", "claim": "c", "url": "u", "excerpt": "짧음"}]
    compact = evidence_store.offload("trace-x", "tech", ev)
    assert len(compact[0]["excerpt"]) == STATE_EXCERPT_CHARS and compact[0]["excerpt_ref"].startswith("trace-x/tech/")
    assert "excerpt_ref" not in compact[1]  # 이미 짧으면 그대로
    assert evidence_store.hydrate(compact)[0]["excerpt"] == full


def test_state_keeps_compact_evidence_but_readers_get_full_text(run_graph):
    llm = FakeLLM()
    state = run_graph(llm)
    assert state["evidence"] and all(len(ev["excerpt"]) <= STATE_EXCERPT_CHARS for ev in state["evidence"])
    assert any(ev.get("excerpt_ref") for ev in state["evidence"])
    # 보고서 에이전트의 인용 카탈로그에는 300자보다 긴 원문(최대 900자)이 들어간다
    excerpts = re.findall(r"'excerpt': '([^']*)'", llm.prompts["report"][-1])
    assert excerpts and max(map(len, excerpts)) > STATE_EXCERPT_CHARS
    # 품질 평가 Judge의 인용 발췌도 원문이다
    snippets = cited_snippets(state)
    assert snippets and max(len(s["excerpt"]) for s in snippets) > STATE_EXCERPT_CHARS


def test_missing_store_is_flagged_not_silent(run_graph, isolated_outputs):
    import shutil
    from kv_eval.evaluation.quality import quality_evaluator_node
    llm = FakeLLM()
    state = run_graph(llm)
    assert evidence_store.missing_full_text(state["evidence"]) == 0
    shutil.rmtree(isolated_outputs / "cache")  # 원문 저장소 삭제(새 clone과 같은 상황)
    assert evidence_store.missing_full_text(state["evidence"]) > 0
    result = quality_evaluator_node(state, llm)["eval_result"]
    assert result["evidence_store"]["cited_missing_full_text"] > 0 and result["warnings"]
    assert "TRUNCATED" in llm.prompts["judge"][-1]  # Judge에게 축약 사실을 알린다
    assert all(s["excerpt_truncated"] for s in cited_snippets(state))


def test_report_only_fails_fast_without_store(isolated_outputs, monkeypatch, run_graph):
    import importlib.util
    import json
    import shutil
    from pathlib import Path
    import pytest
    state = run_graph(FakeLLM())
    out = isolated_outputs / "outputs"
    (out / "final_state.json").write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
    shutil.rmtree(isolated_outputs / "cache")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("kv_app_ro", root / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    with pytest.raises(RuntimeError, match="--report-only 불가"):
        app.report_only()


def test_rehydrated_seed_is_reoffloaded():
    full = "나" * 900
    first = evidence_store.offload("trace-a", "tech", [{"source_id": "s", "claim": "c", "page": 1, "excerpt": full}])
    hydrated = evidence_store.hydrate(first)
    again = evidence_store.offload("trace-b", "seed", hydrated)
    assert again[0]["excerpt_ref"].startswith("trace-b/seed/") and evidence_store.hydrate(again)[0]["excerpt"] == full
