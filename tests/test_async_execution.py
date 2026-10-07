"""비동기 노드, 에이전트 동시 실행 상한, 임베딩 캐시 테스트."""
from __future__ import annotations
import inspect
import threading
import time

import pytest

from fakes import FakeLLM
from kv_eval import observability
from kv_eval.agents import domain, market, stakeholder, technology
from kv_eval.graph import WORKER_NODES, build_graph
from kv_eval.observability import run_config

PERSPECTIVE_FNS = ((technology, "technology_node"), (market, "market_node"),
                   (stakeholder, "stakeholder_node"), (domain, "domain_node"))


def _track_agent_concurrency(monkeypatch) -> dict:
    """관점 에이전트의 최대 동시 실행 수를 잰다."""
    seen = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def track(fn):
        def inner(*args, **kwargs):
            with lock:
                seen["active"] += 1
                seen["peak"] = max(seen["peak"], seen["active"])
            try:
                time.sleep(0.05)  # 동시 실행이 허용되면 여기서 겹친다
                return fn(*args, **kwargs)
            finally:
                with lock:
                    seen["active"] -= 1
        return inner

    for module, attr in PERSPECTIVE_FNS:
        monkeypatch.setattr(module, attr, track(getattr(module, attr)))
    return seen


def test_worker_nodes_are_async():
    graph = build_graph(None, None, None)
    for name in WORKER_NODES:
        assert inspect.iscoroutinefunction(graph.builder.nodes[name].runnable.afunc), name


def test_perspective_agents_run_one_at_a_time_by_default(monkeypatch, run_graph):
    assert run_config("t")["max_concurrency"] == 1
    seen = _track_agent_concurrency(monkeypatch)
    state = run_graph(FakeLLM())
    assert state["status"] == "completed"
    assert seen["peak"] == 1  # 4관점을 한 번에 dispatch해도 하나씩 실행


def test_concurrency_setting_is_what_serializes_agents(monkeypatch, run_graph):
    """상한을 풀면 겹치므로 위 테스트의 peak==1은 설정 때문이다."""
    monkeypatch.setattr(observability, "AGENT_CONCURRENCY", 4)
    seen = _track_agent_concurrency(monkeypatch)
    run_graph(FakeLLM())
    assert seen["peak"] > 1


def test_document_embeddings_are_computed_once_and_reused(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("faiss")
    from kv_eval.rag.index import load_or_embed

    class CountingModel:
        calls = 0

        def encode(self, texts, **kwargs):
            CountingModel.calls += 1
            return np.arange(1, len(texts) * 4 + 1, dtype="float32").reshape(len(texts), 4)

    chunks = [{"chunk_id": f"c{i}", "text": f"chunk {i}"} for i in range(3)]
    first, cached_first = load_or_embed(chunks, CountingModel(), tmp_path)
    second, cached_second = load_or_embed(chunks, CountingModel(), tmp_path)
    assert (cached_first, cached_second) == (False, True)
    assert CountingModel.calls == 1 and np.allclose(first, second)
    assert np.allclose(np.linalg.norm(second, axis=1), 1.0)  # 정규화된 벡터를 저장한다
    # 청크가 바뀌면 다시 임베딩한다
    _, cached_changed = load_or_embed([*chunks, {"chunk_id": "c3", "text": "new"}], CountingModel(), tmp_path)
    assert not cached_changed and CountingModel.calls == 2
