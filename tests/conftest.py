from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """결정 로그·RAG 캐시·체크포인트를 테스트 임시 폴더로 보낸다(저장소 outputs/ 오염 방지)."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MANIFEST_PATH", str(ROOT / "data" / "manifest.json"))
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    return tmp_path


@pytest.fixture
def run_graph():
    from fakes import FakeRAG, FakeWeb
    from kv_eval.graph import build_graph
    from kv_eval.observability import new_trace_id, run_config
    from kv_eval.state import initial_state

    def _run(llm, web=None, rag=None, policy=None):
        trace_id = new_trace_id()
        web = web or FakeWeb()
        graph = build_graph(rag or FakeRAG(), web, llm, policy=policy)
        state = graph.invoke(initial_state("테스트 질의", trace_id), config=run_config(trace_id))
        return state

    return _run
